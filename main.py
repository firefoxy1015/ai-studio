import asyncio
import base64
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bs4 import BeautifulSoup

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── IDEXX Neo credentials (from Render env vars) ─────────────────────────────
NEO_COMPANY = os.environ.get("NEO_COMPANY", "8890")
NEO_USER    = os.environ.get("NEO_USER", "")
NEO_PASS    = os.environ.get("NEO_PASS", "")
NEO_BASE    = "https://us.idexxneo.com"

DATA999_KEY = "sk-37b060cd778ee075ac3388fe421c6df1cc367f591238195c"
DATA999_BASE = "https://api.ai6700.com"
BYTEFOR_KEY = "sk-4afb143170cc4bb996d7533939efce4c8c02a3a5070e493d"
BYTEFOR_BASE = "https://open.bytefor.com"
DEEPWL_KEY = "sk-hUviZm3xQzam0EaaA9622c041aA249CbB4924c929c9805Aa"
DEEPWL_BASE = "https://zx1.deepwl.net"

CLAUDE_MODELS = {
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101",
}
DATA999_ONLY_MODELS = {"grok-4.2", "grok-4.2-image", "doubao-seed-2-0-pro-260215"}

LINGKEAI_BASE = "https://php.lingkeai.vip"
LINGKEAI_SESSION_TOKEN = "e5b7ae5474930aaba74e50025f263888"
LINGKEAI_USER_ID = "9011036"
LINGKEAI_S6 = "5Mmhj96mnRi@aHxcFFtA-kWRWilQGwVK"
LINGKEAI_MODEL_IDS = {
    # Grok
    "grok-4.2": 94,
    "grok-4.2-image": 48,
    "grok-video-3": 31,
    "grok-video-3-plus": 63,
    # Claude
    "claude-opus-4-7": 90,
    "claude-sonnet-4-6": 38,
    "claude-opus-4-6": 33,
    # Gemini
    "gemini-3.1-pro-preview": 39,
    "gemini-3-pro-preview": 5,
    # GPT
    "gpt-5.4": 58,
    "gpt-5.4-mini": 92,
    "gpt-5.4-nano": 91,
    "gpt-5.4-xhigh": 71,
    # Other
    "doubao-seed-2-0-pro-260215": 40,
    "qwen3.6-plus": 85,
    "MiniMax-M2.7": 82,
}

# Reasoning models need higher max_tokens (thinking chain exhausts budget)
REASONING_MODELS = {
    "gpt-5.4-xhigh", "deepseek-r1", "deepseek-r2",
    "qwq", "qwen3.6-plus", "MiniMax-M2.7",
}

GEMINI_MODELS = {
    "gemini-3.1-pro-preview", "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview",
}
SINGLE_IMG_MODELS = {"sora-2-all", "grok-video-3", "grok-video-3-plus"}
SORA2_MODEL = "sora-2"

# Global HTTP client — connection pooling avoids TCP handshake overhead per request
_http: httpx.AsyncClient | None = None

def http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=120,
            verify=False,  # some upstream APIs have expired certs
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
        )
    return _http


def _encode_lingkeai_token() -> str:
    timestamp = int(time.time())
    n = (LINGKEAI_SESSION_TOKEN + "|" + str(timestamp)).encode("utf-8")
    key = LINGKEAI_S6.encode("utf-8")
    xored = bytes([n[i] ^ key[i % len(key)] for i in range(len(n))])
    return base64.b64encode(xored).decode("ascii")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[str] = None
    web_search: bool = False
    enable_thinking: bool = False
    priority: str = "成功率优先"  # "成功率优先" | "价格优先" — lingkeai 渠道分组策略


class GenerateRequest(BaseModel):
    source: str = "data999"
    model: str
    prompt: str
    params: Optional[Dict[str, Any]] = {}


class DirectorRequest(BaseModel):
    role_id: int
    raw_prompt: str
    role_name: str
    role_system: str


class AgentPlanRequest(BaseModel):
    agent_type: str
    text: str
    images: List[str] = []


AGENT_PLAN_PROMPTS: Dict[str, str] = {
    "ecommerce": (
        "你是专业电商产品图策划师。用户提供了商品信息，参考图URL已在用户输入中（如有）。\n"
        "规划6张不同类型的电商图，每张有具体的英文生图提示词：\n"
        "1. 主图白底：纯白背景专业商品照\n"
        "2. 主图场景：生活场景中的商品\n"
        "3. 细节特写：商品局部/材质细节\n"
        "4. 使用场景：用户正在使用的场景\n"
        "5. 促销海报：有视觉冲击力的营销图\n"
        "6. 社媒方图：适合Instagram/小红书的风格图\n\n"
        "直接输出JSON数组（6个元素），格式：\n"
        '[{"label":"主图白底","prompt":"professional product photography on pure white background...","model":"wan2.7-image","category":"image","params":{"quality":"standard","size":"1:1"}},...]\n'
        "prompt全英文60-100字，专业具体。直接输出JSON数组，不要任何其他文字。"
    ),
    "general_image": (
        "你是创意图片策划师。根据用户的参考图和描述，规划6张风格各异但主题一致的系列图。\n"
        "从不同角度/风格/情绪出发，形成完整视觉叙事：全景氛围、主体突出、细节特写、不同光线、不同情绪、创意角度。\n\n"
        "直接输出JSON数组（6个元素）：\n"
        '[{"label":"全景氛围","prompt":"wide cinematic shot...","model":"wan2.7-image","category":"image","params":{"quality":"standard","size":"16:9"}},...]\n'
        "prompt全英文60-100字，风格差异明显。直接输出JSON数组，不要其他文字。"
    ),
    "manga_s2": (
        "你是漫剧导演。根据用户提供的参考图片描述和主题，创作一个5幕漫剧短片脚本。\n"
        "叙事弧：开场→发展→冲突→高潮→结局。镜头语言丰富，情感递进有张力。\n\n"
        "直接输出JSON数组（5个元素）：\n"
        '[{"label":"第1幕 开场","prompt":"Cinematic wide establishing shot, golden hour light...slow dolly push-in, melancholic atmosphere, 4K cinematic","model":"veo3.1-lite","category":"video","params":{"enhance_prompt":true,"quality":"4k"}},...]\n'
        "prompt全英文60-100字，镜头感强。直接输出JSON数组，不要其他文字。"
    ),
    "manga_plot": (
        "你是漫剧导演。将用户提供的剧情文案改编为5幕漫剧视频脚本。\n"
        "忠于情感主线，文字转化为具体画面描述。5幕有完整叙事结构，每幕有电影感。\n\n"
        "直接输出JSON数组（5个元素）：\n"
        '[{"label":"第1幕 开场","prompt":"Cinematic opening...","model":"veo3.1-lite","category":"video","params":{"enhance_prompt":true,"quality":"4k"}},...]\n'
        "prompt全英文60-100字。直接输出JSON数组，不要其他文字。"
    ),
    "manga_narration": (
        "你是抖音解说视频导演。将用户内容做成3-5幕解说风格短视频，每幕包含：\n"
        "1. 英文画面提示词（用于生成视频）\n"
        "2. 中文解说文字（20字内，口语化，有悬念钩子）\n"
        "节奏快、视觉冲击强。\n\n"
        "直接输出JSON数组（3-5个元素）：\n"
        '[{"label":"第1幕 钩子","prompt":"dramatic extreme close-up...","narration":"你知道吗，这个秘密改变了一切...","model":"veo3.1-lite","category":"video","params":{"enhance_prompt":true,"quality":"4k"}},...]\n'
        "直接输出JSON数组，不要其他文字。"
    ),
    "multi_frame": (
        "你是视频剪辑师。用户上传了多张图片，你需要为每张图片生成一个独立视频片段提示词，串联成连贯长视频。\n"
        "分析整体风格和氛围，每段镜头语言有变化（推/拉/摇/移），前后连贯统一。\n"
        "输出的JSON数组元素数量必须与用户上传图片数量完全相同。\n\n"
        "直接输出JSON数组：\n"
        '[{"label":"片段1","prompt":"slow cinematic push-in...","model":"veo3.1-lite","category":"video","params":{"enhance_prompt":true,"quality":"4k"}},...]\n'
        "prompt全英文60-100字，镜头动作明确。直接输出JSON数组，不要其他文字。"
    ),
}


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("index.html").read_text(encoding="utf-8")



def _lingkeai_body(req: ChatRequest, model_id: int) -> tuple[dict, dict]:
    parts = []
    if req.system:
        parts.append(f"[System: {req.system}]")
    for m in req.messages:
        role = "Human" if m.role == "user" else "Assistant"
        parts.append(f"{role}: {m.content}")
    user_msg = "\n".join(parts)
    group_id = f"group_{LINGKEAI_USER_ID}_{int(time.time() * 1000)}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "token": _encode_lingkeai_token(),
    }
    max_tokens = 1500 if req.model in REASONING_MODELS else 8192
    strategy = req.priority if req.priority in ("成功率优先", "价格优先") else "成功率优先"
    body = {
        "模型id": model_id,
        "用户消息": user_msg,
        "渠道分组策略": strategy,
        "对话组id": group_id,
        "生成参数": {
            "web_search": req.web_search,
            "enable_thinking": req.enable_thinking,
            "max_tokens": max_tokens,
        },
    }
    return headers, body


async def _stream_lingkeai(req: ChatRequest):
    model_id = LINGKEAI_MODEL_IDS.get(req.model)
    if not model_id:
        raise Exception(f"No lingkeai model ID for {req.model}")
    headers, body = _lingkeai_body(req, model_id)
    got_text = False
    async with http().stream("POST", f"{LINGKEAI_BASE}/moxing/tongyirukouchat",
                              headers=headers, json=body) as r:
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            try:
                d = json.loads(s)
                if d.get("type") == "error":
                    raise Exception(d.get("message", "lingkeai error"))
                text = ""
                if d.get("type") == "content" and d.get("content"):
                    text = d["content"]
                elif d.get("choices"):
                    text = d["choices"][0].get("delta", {}).get("content", "")
                if text:
                    got_text = True
                    yield f"data: {json.dumps({'text': text})}\n\n"
            except json.JSONDecodeError:
                pass
    if not got_text:
        raise Exception("lingkeai returned empty response")
    yield "data: [DONE]\n\n"


async def _stream_lingkeai_only(req: ChatRequest):
    """Call lingkeai only — no fallback to DATA999 key."""
    model_id = LINGKEAI_MODEL_IDS.get(req.model)
    if not model_id:
        yield f"data: {json.dumps({'text': f'模型 {req.model} 暂不支持，请联系管理员'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    async for chunk in _stream_lingkeai(req):
        yield chunk


# ── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    gen = _stream_lingkeai_only(req)
    return StreamingResponse(
        gen, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Sync wrapper — lingkeai only, no DATA999 key."""
    model_id = LINGKEAI_MODEL_IDS.get(req.model)
    if not model_id:
        raise HTTPException(400, detail=f"模型 {req.model} 暂不支持")
    full = ""
    async for chunk in _stream_lingkeai(req):
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
            try:
                full += json.loads(chunk[6:].strip()).get("text", "")
            except Exception:
                pass
    if not full:
        raise HTTPException(503, detail=f"lingkeai 返回空响应")
    return {"text": full}


# ── Multi-model collaboration (data999 大模型 旗舰功能) ──
class MultiChatRequest(BaseModel):
    question: str
    models: List[str]                    # 2-4 model IDs to fan out to
    summarizer: str = "claude-opus-4-7"  # judge model
    system: Optional[str] = None
    history: List[Message] = []          # prior turns (optional)
    priority: str = "成功率优先"          # "成功率优先" | "价格优先"


async def _one_shot_lingkeai(model_key: str, messages: List[Message], system: Optional[str], priority: str = "成功率优先") -> tuple[str, str]:
    """Single-call helper for parallel fan-out. Returns (model_key, text or error)."""
    try:
        model_id = LINGKEAI_MODEL_IDS.get(model_key)
        if not model_id:
            return model_key, f"[模型 {model_key} 不支持]"
        req = ChatRequest(model=model_key, messages=messages, system=system,
                          web_search=False, enable_thinking=False, priority=priority)
        full = ""
        async for chunk in _stream_lingkeai(req):
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    full += json.loads(chunk[6:].strip()).get("text", "")
                except Exception:
                    pass
        return model_key, full or "[空响应]"
    except Exception as e:
        return model_key, f"[错误: {str(e)[:200]}]"


@app.post("/api/chat/multi")
async def chat_multi(req: MultiChatRequest):
    """多模型协作：N 个模型并行回答 → 第 N+1 个模型综合总结。"""
    import asyncio as _asyncio
    if len(req.models) < 2:
        raise HTTPException(400, detail="多模型协作至少需要 2 个模型")
    if len(req.models) > 4:
        raise HTTPException(400, detail="最多 4 个模型并行")
    # Build conversation messages (history + current question)
    msgs = list(req.history) + [Message(role="user", content=req.question)]
    # Fan out in parallel
    results = await _asyncio.gather(
        *[_one_shot_lingkeai(m, msgs, req.system, req.priority) for m in req.models],
        return_exceptions=False,
    )
    responses = {k: v for (k, v) in results}
    # Build summarizer prompt
    parts = [f"用户问题：{req.question}\n\n以下是 {len(req.models)} 个不同模型对该问题的回答："]
    for i, (k, v) in enumerate(results, 1):
        parts.append(f"\n──── 模型 {i} ({k}) 的回答 ────\n{v}")
    parts.append(
        "\n\n请你作为综合分析专家，评估这些回答的优缺点，"
        "提取每个模型的独到见解，去除错误或矛盾之处，"
        "最终给出一个最全面、最准确、最实用的综合答案。"
        "用清晰的结构化 Markdown 输出。"
    )
    summarizer_prompt = "".join(parts)
    summarizer_req = ChatRequest(
        model=req.summarizer,
        messages=[Message(role="user", content=summarizer_prompt)],
        system="你是综合分析专家，擅长从多个 AI 模型的回答中提炼最佳答案。",
        web_search=False, enable_thinking=False, priority=req.priority,
    )
    summary = ""
    try:
        async for chunk in _stream_lingkeai(summarizer_req):
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    summary += json.loads(chunk[6:].strip()).get("text", "")
                except Exception:
                    pass
    except Exception as e:
        summary = f"[总结失败: {str(e)[:200]}]"
    return {"responses": responses, "summary": summary or "[空响应]", "summarizer": req.summarizer}


_DIRECTOR_SHOTS = [
    "极端特写", "低角度仰拍", "鸟瞰俯拍", "荷兰角倾斜构图", "跟拍运动镜头",
    "隐藏摄影机偷拍感", "长镜头一镜到底", "分裂画面", "主观第一人称视角", "旋转360度环绕",
]
_DIRECTOR_LIGHT = [
    "黄金时刻逆光", "蓝调时分冷光", "霓虹灯反射湿地", "单一硬光强阴影",
    "烛光温暖摇曳", "日光灯冷白刺眼", "月光冷蓝穿云", "火光跳动暖橙",
    "暗房红灯", "窗格光影斑驳",
]
_DIRECTOR_MOOD = [
    "压抑到爆发", "平静下暗涌危机", "荒诞幽默", "原始本能冲动",
    "孤独但不悲伤", "狂喜接近疯狂", "怀旧但已不可回", "冷酷客观无情感",
    "神秘仪式感", "疲惫但仍燃烧",
]
_DIRECTOR_TEXTURE = [
    "胶片颗粒感强", "过曝高对比", "去饱和近单色", "水彩晕染柔边",
    "监控画质粗糙", "8mm家庭录像", "数字故障glitch", "超现实油画质感",
    "高速摄影极慢动作", "延时摄影快进",
]
_DIRECTOR_STRUCTURE = [
    "从结局开始倒叙", "聚焦一个无关紧要的细节来讲整件事",
    "不出现主体，只拍反应和环境", "用隐喻替代直接描述，完全不提字面意思",
    "两个不相关场景平行剪辑制造对比", "只描述声音和光，不描述人",
    "从微观局部慢慢拉远到宏观", "时间静止，只有一个元素在动",
]


@app.post("/api/director")
async def director_endpoint(req: DirectorRequest):
    """Director modal — embed all instructions in the user message to bypass lingkeai's [System:] ignore bug."""
    # Pick random creative constraints — different every call
    shot = random.choice(_DIRECTOR_SHOTS)
    light = random.choice(_DIRECTOR_LIGHT)
    mood = random.choice(_DIRECTOR_MOOD)
    texture = random.choice(_DIRECTOR_TEXTURE)
    structure = random.choice(_DIRECTOR_STRUCTURE)
    seed = random.randint(10000, 99999)

    combined = (
        f"你现在是【{req.role_name}】这个角色。{req.role_system}\n\n"
        f"用户的原始想法：「{req.raw_prompt}」\n\n"
        f"本次随机创作种子 #{seed}，你必须从以下随机维度出发来诠释这个想法，"
        f"这些维度每次都不同，你的输出必须完全反映它们：\n"
        f"• 镜头语言：{shot}\n"
        f"• 光线氛围：{light}\n"
        f"• 情绪基调：{mood}\n"
        f"• 画面质感：{texture}\n"
        f"• 叙事结构：{structure}\n\n"
        f"以上是强制约束，不能忽略。在这些约束框架内，用【{req.role_name}】独特的导演个性和美学偏好来创作。\n"
        f"禁止使用任何通用模板句式（如'在X的光线下，Y正在做Z'这类格式）。\n\n"
        "直接输出纯JSON，不要任何markdown、代码块或解释：\n"
        '{"explanation":"以你角色的口吻，说出你这次的创作切入点和为什么选这个角度，2-3句，要有个性",'
        '"cnPrompt":"中文视频提示词，必须融入以上所有随机维度，60-100字，句式要多变",'
        '"enPrompt":"English video prompt that reflects all the above random dimensions and your directorial voice, 60-100 words, varied sentence structure"}'
    )
    group_id = f"group_{LINGKEAI_USER_ID}_{int(time.time() * 1000)}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "token": _encode_lingkeai_token(),
    }
    body = {
        "模型id": 38,  # claude-sonnet-4-6 — follows embedded instructions reliably
        "用户消息": combined,
        "渠道分组策略": "成功率优先",
        "对话组id": group_id,
        "生成参数": {"web_search": False, "enable_thinking": False, "max_tokens": 2000},
    }

    async def _stream():
        got_text = False
        try:
            async with http().stream("POST", f"{LINGKEAI_BASE}/moxing/tongyirukouchat",
                                      headers=headers, json=body) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'HTTP {r.status_code}'})}\n\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    s = line[5:].strip()
                    try:
                        d = json.loads(s)
                        if d.get("type") == "error":
                            yield f"data: {json.dumps({'error': d.get('message', 'lingkeai error')})}\n\n"
                            return
                        text = ""
                        if d.get("type") == "content" and d.get("content"):
                            text = d["content"]
                        elif d.get("choices"):
                            text = d["choices"][0].get("delta", {}).get("content", "")
                        if text:
                            got_text = True
                            yield f"data: {json.dumps({'text': text})}\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        if not got_text:
            yield f"data: {json.dumps({'error': 'empty response'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class MangaStageRequest(BaseModel):
    stage: str
    context: Dict[str, Any] = {}
    images: List[str] = []
    feedback: str = ""
    settings: Dict[str, Any] = {}


MANGA_STAGE_PROMPTS: Dict[str, str] = {
    "intent": (
        "你是AI漫剧S2.0的创作助手。深度分析用户的创意描述，准确理解创作意图。\n\n"
        "直接返回JSON（不要markdown代码块）：\n"
        '{"intent_type":"narrative（从零创作故事）或adapt（改编已有素材）",'
        '"core_themes":["主题1","主题2"],"main_characters":["角色描述"],'
        '"setting":"主要场景和环境","visual_style":"推荐画面风格（如：赛博朋克/温馨日常/古风武侠）",'
        '"summary":"3句话总结这个创意的核心魅力和故事潜力，口吻专业有激情"}'
    ),
    "directions": (
        "你是AI漫剧S2.0的创意总监。基于意图分析，构思3个风格鲜明且完全不同的创意方向。\n\n"
        "直接返回JSON数组（不要markdown代码块）：\n"
        '[{"id":1,"title":"方向名称（4-8字，有感染力）","style_tag":"情感基调标签（如：悬疑反转/温情治愈/热血励志）",'
        '"synopsis":"故事走向（3-4句，有起伏有悬念）","highlight":"最大亮点（1句话）",'
        '"visual_keyword":"核心视觉关键词（如：霓虹夜雨/暖黄复古/冷蓝科技）"},'
        '{"id":2,...},{"id":3,...}]\n'
        "三个方向必须有明显差异，覆盖不同情感基调。"
    ),
    "storyboard": (
        "你是AI漫剧S2.0的首席分镜师。基于选定的创意方向，按用户硬性参数（帧数和时长）创作分镜脚本。\n\n"
        "⚠️ 关键一致性要求（每帧 image_prompt 必须 100% 包含）：\n"
        "1. **角色锚定**：每帧用完全相同的角色描述（外观/颜色/品种/服装/特征），不能改一个字\n"
        "2. **画风锚定**：基于意图分析的 visual_style 和方向的 visual_keyword，每帧画风描述完全相同\n"
        "3. **色调锚定**：色温/色调系统每帧统一（如：暖橙阳光 / 冷蓝霓虹 / 复古黄绿）\n\n"
        "示例（橘猫故事）：每帧 image_prompt 都要这样开头：\n"
        "「一只橙色短毛橘猫，圆绿眼，白色胸口和爪尖，温馨日式手绘插画风，暖橙阳光色调」\n"
        "然后再描述这一帧的具体场景/动作/构图。\n\n"
        "直接返回JSON数组（不要markdown代码块），帧数严格按用户参数：\n"
        '[{"shot":1,"title":"分镜标题（5字内）","scene":"场景描述",'
        '"characters":["出境角色"],"action":"主要动作和画面变化",'
        '"dialogue":"台词或旁白（可为空字符串）","camera":"特写/中景/远景/全景",'
        '"duration":5,"image_prompt":"必须以【完全相同的角色+画风+色调】开头，再加该帧场景动作，100-130字",'
        '"video_prompt":"English video prompt（camera movement + scene dynamics + atmosphere，40-60 words）"}]\n'
        "分镜要有完整故事弧：开篇铺垫→发展→冲突→高潮→结尾。帧数和每帧 duration 严格按用户硬性参数。"
    ),
    "plot_storyboard": (
        "你是AI漫剧导演。将用户提供的剧情文案转化为分镜脚本（帧数和时长按用户硬性参数），保持情感主线和故事张力。\n\n"
        "⚠️ 关键一致性要求（每帧 image_prompt 必须 100% 包含完全相同的开头）：\n"
        "1. **角色锚定**：每帧用完全相同的角色描述（外观/服装/特征），不能改一个字\n"
        "2. **画风锚定**：每帧画风描述完全相同（如：电影感写实/温馨手绘插画/赛博朋克）\n"
        "3. **色调锚定**：色温色调每帧统一\n\n"
        "image_prompt 示例开头：「[完全相同的角色描述]，[相同画风]，[相同色调]，」+ 该帧具体场景动作\n\n"
        "直接返回JSON数组（不要markdown代码块），帧数严格按用户参数：\n"
        '[{"shot":1,"title":"分镜标题（5字内）","scene":"场景描述",'
        '"characters":["出境角色"],"action":"主要动作",'
        '"dialogue":"台词或旁白（可为空字符串）","camera":"特写/中景/远景/全景",'
        '"duration":5,"image_prompt":"必须以【完全相同的角色+画风+色调】开头，再加该帧场景，100-130字",'
        '"video_prompt":"English video prompt（40-60 words）"}]\n'
        "忠于原文情感，镜头语言丰富，有电影感。帧数和每帧 duration 严格按用户硬性参数。"
    ),
    "narration_storyboard": (
        "你是抖音解说视频导演。将用户内容拆解为解说视频片段（数量按用户参数），每段配合具体的视觉画面。\n\n"
        "⚠️ 关键一致性要求（每帧 image_prompt 必须 100% 包含相同的开头）：\n"
        "1. **风格锚定**：每帧使用完全相同的视觉风格描述（如：抖音爆款/赛博霓虹/复古胶片）\n"
        "2. **色调锚定**：色温色调每帧统一\n\n"
        "image_prompt 示例开头：「[相同视觉风格]，[相同色调]，」+ 该帧场景\n\n"
        "直接返回JSON数组（不要markdown代码块），片段数严格按用户参数：\n"
        '[{"shot":1,"title":"片段标题（5字内）","scene":"视觉场景描述",'
        '"narration":"解说文字（15-25字，口语化有钩子）","action":"画面主要动作",'
        '"camera":"特写/中景/远景","duration":5,'
        '"image_prompt":"必须以【相同的风格+色调】开头，再加该帧场景，80-110字",'
        '"video_prompt":"English video prompt（40-60 words）"}]\n'
        "解说节奏快、有悬念、视觉冲击强，契合抖音调性。片段数和每段 duration 严格按用户硬性参数。"
    ),
}


@app.post("/api/manga/stage")
async def manga_stage_endpoint(req: MangaStageRequest):
    """Manga S2.0 pipeline — per-stage AI generation with streaming."""
    stage = req.stage
    if stage not in MANGA_STAGE_PROMPTS:
        raise HTTPException(400, f"Unknown manga stage: {stage}")

    sys_instr = MANGA_STAGE_PROMPTS[stage]
    ctx = req.context
    parts: List[str] = []

    if req.feedback:
        parts.append(f"【修改意见】{req.feedback}\n")

    if stage == "intent":
        parts.append(f"用户创意描述：{ctx.get('text', '')}")
        if req.images:
            parts.append(f"上传了{len(req.images)}张参考图，URL：{' | '.join(req.images[:3])}")
    elif stage == "directions":
        parts.append(f"原始创意：{ctx.get('text', '')}")
        parts.append(f"意图分析结果：{json.dumps(ctx.get('intent_result', {}), ensure_ascii=False)}")
        styles = req.settings.get("visualStyles") or []
        if styles:
            parts.append(
                f"⚠ 用户已勾选画面风格偏好：{' / '.join(styles)}。"
                f"3 个方向必须分别围绕这些风格变化，每个方向 visual_keyword 要呼应至少一个用户风格。"
            )
    elif stage in ("storyboard", "plot_storyboard", "narration_storyboard"):
        parts.append(f"原始创意/内容：{ctx.get('text', '')}")
        if ctx.get("selected_direction"):
            parts.append(f"选定创意方向：{json.dumps(ctx['selected_direction'], ensure_ascii=False)}")
        if req.images:
            parts.append(f"参考图数量：{len(req.images)}张")
        # User-configurable shot count + duration + style overrides
        shot_count = int(req.settings.get("shotCount", 5) or 5)
        shot_count = max(3, min(10, shot_count))
        default_dur = int(req.settings.get("defaultDuration", 5) or 5)
        default_dur = max(3, min(15, default_dur))
        styles = req.settings.get("visualStyles") or []
        parts.append(
            f"⚙ 用户硬性参数（必须严格遵守）：\n"
            f"- 必须生成 **{shot_count} 帧** 分镜（不多不少，shot 字段从 1 编号到 {shot_count}）\n"
            f"- 每帧的 duration 字段统一设为 {default_dur}（用户已选默认时长）"
        )
        if styles:
            parts.append(f"- 用户额外指定画面风格关键词：{' / '.join(styles)} —— 必须融入每帧 image_prompt 的画风锚定中")

    combined = f"{sys_instr}\n\n[用户输入]\n" + "\n".join(parts)

    group_id = f"group_{LINGKEAI_USER_ID}_{int(time.time() * 1000)}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "token": _encode_lingkeai_token(),
    }
    # Storyboard stages need more tokens — 5-7 shots × ~400 chars each + JSON overhead
    storyboard_stages = ("storyboard", "plot_storyboard", "narration_storyboard")
    max_tok = 8000 if stage in storyboard_stages else 4000
    body = {
        "模型id": 38,
        "用户消息": combined,
        "渠道分组策略": "成功率优先",
        "对话组id": group_id,
        "生成参数": {"web_search": False, "enable_thinking": False, "max_tokens": max_tok},
    }

    async def _stream():
        got_text = False
        try:
            async with http().stream("POST", f"{LINGKEAI_BASE}/moxing/tongyirukouchat",
                                      headers=headers, json=body) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'HTTP {r.status_code}'})}\n\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    s = line[5:].strip()
                    try:
                        d = json.loads(s)
                        if d.get("type") == "error":
                            yield f"data: {json.dumps({'error': d.get('message', 'error')})}\n\n"
                            return
                        text = ""
                        if d.get("type") == "content" and d.get("content"):
                            text = d["content"]
                        elif d.get("choices"):
                            text = d["choices"][0].get("delta", {}).get("content", "")
                        if text:
                            got_text = True
                            yield f"data: {json.dumps({'text': text})}\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        if not got_text:
            yield f"data: {json.dumps({'error': 'empty response'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent/plan")
async def agent_plan_endpoint(req: AgentPlanRequest):
    """Agent planning — Claude generates a JSON task list for the given agent type."""
    prompt_template = AGENT_PLAN_PROMPTS.get(req.agent_type)
    if not prompt_template:
        raise HTTPException(400, f"Unknown agent type: {req.agent_type}")

    img_section = ""
    if req.images:
        img_section = f"\n用户上传了{len(req.images)}张参考图片：{' | '.join(req.images)}\n"

    combined = f"{prompt_template}{img_section}\n用户输入：{req.text}"

    group_id = f"group_{LINGKEAI_USER_ID}_{int(time.time() * 1000)}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "token": _encode_lingkeai_token(),
    }
    body = {
        "模型id": 38,
        "用户消息": combined,
        "渠道分组策略": "成功率优先",
        "对话组id": group_id,
        "生成参数": {"web_search": False, "enable_thinking": False, "max_tokens": 4000},
    }

    async def _stream():
        got_text = False
        try:
            async with http().stream("POST", f"{LINGKEAI_BASE}/moxing/tongyirukouchat",
                                      headers=headers, json=body) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'HTTP {r.status_code}'})}\n\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    s = line[5:].strip()
                    try:
                        d = json.loads(s)
                        if d.get("type") == "error":
                            yield f"data: {json.dumps({'error': d.get('message', 'error')})}\n\n"
                            return
                        text = ""
                        if d.get("type") == "content" and d.get("content"):
                            text = d["content"]
                        elif d.get("choices"):
                            text = d["choices"][0].get("delta", {}).get("content", "")
                        if text:
                            got_text = True
                            yield f"data: {json.dumps({'text': text})}\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        if not got_text:
            yield f"data: {json.dumps({'error': 'empty response'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    params = dict(req.params or {})

    if req.source == "deepwl":
        r = await http().post(
            f"{DEEPWL_BASE}/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPWL_KEY}", "Content-Type": "application/json"},
            json={"model": req.model, "messages": [{"role": "user", "content": req.prompt}], "stream": False},
            timeout=300,
        )
        d = r.json()
        print(f"[deepwl] status={r.status_code} model={req.model} resp_keys={list(d.keys())} error={d.get('error')} content_preview={str(d.get('choices',[{}])[0].get('message',{}).get('content',''))[:200]}", flush=True)
        if d.get("error"):
            raise HTTPException(400, detail=d["error"].get("message", str(d["error"])))
        content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        video_exts = r'\.(?:mp4|webm|mov|m3u8)'
        urls = re.findall(rf'https?://[^\s\)\"\']+{video_exts}[^\s\)\"\']*', content)
        if not urls:
            urls = re.findall(r'(?<!!)\[.*?\]\((https?://[^\)]+)\)', content)
        if not urls:
            raise HTTPException(400, detail=content or "deepwl 未返回媒体链接")
        return {"result_urls": urls, "source": "deepwl"}

    if req.source == "bytefor":
        r = await http().post(
            f"{BYTEFOR_BASE}/api/v1/generate",
            headers={"Authorization": f"Bearer {BYTEFOR_KEY}", "Content-Type": "application/json"},
            json={"prompt": req.prompt, "model": req.model, **params},
            timeout=35,
        )
        d = r.json()
        if d.get("code") != 0:
            raise HTTPException(400, detail=d.get("msg", str(d)))
        return {"task_id": d["data"]["taskCode"], "source": "bytefor"}

    if "images" in params:
        imgs = params["images"]
        if isinstance(imgs, list):
            if req.model in SINGLE_IMG_MODELS:
                params["images"] = imgs[0] if imgs else ""
            elif req.model == SORA2_MODEL:
                params["input_reference"] = imgs[0] if imgs else ""
                del params["images"]

    r = await http().post(
        f"{DATA999_BASE}/v1/media/generate",
        headers={"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"},
        json={"model": req.model, "prompt": req.prompt, "params": params, "count": 1},
        timeout=35,
    )
    d = r.json()
    if d.get("code") != 200:
        raise HTTPException(400, detail=d.get("message", str(d)))

    task_id = None
    for v in d.get("data", {}).values():
        if isinstance(v, list) and v:
            task_id = str(v[0]).strip()
            break
        if isinstance(v, (int, str)) and str(v).strip():
            task_id = str(v).strip()
            break
    if not task_id:
        raise HTTPException(500, "无法提取 task_id")
    return {"task_id": task_id, "source": "data999"}


@app.get("/api/status/{task_id}")
async def status(task_id: str, source: str = "data999"):
    if source == "bytefor":
        r = await http().get(
            f"{BYTEFOR_BASE}/api/v1/task/{task_id}",
            headers={"Authorization": f"Bearer {BYTEFOR_KEY}"},
            timeout=20,
        )
        d = r.json().get("data", {})
        st = d.get("status", "")
        files = d.get("files", [])
        hd = next((f["fileUrl"] for f in files if f.get("fileType") == "video_hd"), None)
        sd = next((f["fileUrl"] for f in files if f.get("fileType") == "video_sd"), None)
        return {
            "is_final": st in ("completed", "failed"),
            "status": st,
            "progress": d.get("progress", 0),
            "progress_text": d.get("progressText", ""),
            "result_urls": [hd or sd] if (hd or sd) else [],
            "error": d.get("errorMsg") if st == "failed" else None,
        }

    r = await http().get(
        f"{DATA999_BASE}/v1/skills/task-status?task_id={task_id}",
        headers={"Authorization": f"Bearer {DATA999_KEY}"},
        timeout=20,
    )
    d = r.json()
    urls = list(d.get("result_urls") or [])
    if not urls and d.get("result_url"):
        urls = [d["result_url"]]
    return {
        "is_final": d.get("is_final", False),
        "status": d.get("status", ""),
        "progress": d.get("progress", 0),
        "result_urls": urls,
        "error": d.get("error"),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    r = await http().post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
        timeout=60,
    )
    url = r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    return {"url": url, "name": file.filename}


def _ffmpeg_bin():
    """Return path to bundled ffmpeg binary (works on Render without apt-get)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fallback to system ffmpeg


async def _upload_to_tmpfiles(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes to tmpfiles.org and return public download URL."""
    r = await http().post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (filename, data, content_type)},
        timeout=120,
    )
    return r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")


class LastFrameRequest(BaseModel):
    video_url: str


@app.post("/api/manga/last-frame")
async def manga_last_frame(req: LastFrameRequest):
    """Download a video and extract its last frame as a JPG, returning a public URL."""
    import subprocess, tempfile, os as _os
    try:
        async with http().stream("GET", req.video_url, timeout=120) as resp:
            if resp.status_code != 200:
                raise HTTPException(400, f"Cannot fetch video: HTTP {resp.status_code}")
            video_bytes = b""
            async for chunk in resp.aiter_bytes():
                video_bytes += chunk

        with tempfile.TemporaryDirectory() as td:
            vpath = _os.path.join(td, "in.mp4")
            ipath = _os.path.join(td, "out.jpg")
            with open(vpath, "wb") as f:
                f.write(video_bytes)
            # Extract the very last frame
            cmd = [
                _ffmpeg_bin(), "-y", "-sseof", "-0.5", "-i", vpath,
                "-vsync", "0", "-q:v", "2", "-update", "1", ipath,
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            if proc.returncode != 0 or not _os.path.exists(ipath):
                raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')[:300]}")
            with open(ipath, "rb") as f:
                img_bytes = f.read()

        img_url = await _upload_to_tmpfiles(img_bytes, "last_frame.jpg", "image/jpeg")
        return {"url": img_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"last-frame error: {e}")


class ConcatRequest(BaseModel):
    video_urls: List[str]


@app.post("/api/manga/concat")
async def manga_concat(req: ConcatRequest):
    """Download multiple video clips and concatenate them into a single MP4.

    Strategy: download → re-encode each to identical params → concat demuxer
    (concat filter would also work but uses more RAM; demuxer with identical
    params is the most reliable approach across mixed CDN sources).
    """
    import subprocess, tempfile, os as _os
    if not req.video_urls or len(req.video_urls) < 2:
        raise HTTPException(400, "Need at least 2 video URLs")

    ffbin = _ffmpeg_bin()
    debug_log: List[str] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            # 1) Download every clip
            local_paths: List[str] = []
            for i, url in enumerate(req.video_urls):
                p = _os.path.join(td, f"clip_{i:03d}.mp4")
                try:
                    async with http().stream(
                        "GET", url, timeout=300,
                        follow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0"},
                    ) as resp:
                        if resp.status_code != 200:
                            raise HTTPException(400, f"clip {i} download failed: HTTP {resp.status_code} from {url[:80]}")
                        size = 0
                        with open(p, "wb") as f:
                            async for chunk in resp.aiter_bytes():
                                f.write(chunk)
                                size += len(chunk)
                        debug_log.append(f"clip{i}: {size} bytes from {url[:60]}")
                except httpx.HTTPError as he:
                    raise HTTPException(400, f"clip {i} network error: {he}")
                if _os.path.getsize(p) < 1024:
                    raise HTTPException(400, f"clip {i} too small ({_os.path.getsize(p)} bytes), likely a broken URL")
                local_paths.append(p)

            # 2) Re-encode every clip to identical params.
            # Render free-tier CPU is slow → use ultrafast + 720p + crf 28 to fit in time budget.
            # Target: 720x1280 vertical (most common for manga); pad shorter side, crop longer side.
            normalized: List[str] = []
            for i, src in enumerate(local_paths):
                dst = _os.path.join(td, f"norm_{i:03d}.mp4")
                cmd = [
                    ffbin, "-y", "-i", src,
                    # Scale longest edge to 1280, pad to 720x1280 (vertical) or letterbox naturally.
                    # Using a uniform output box; mismatched aspect ratios get letterboxed (black bars).
                    "-vf", "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease,"
                           "pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1",
                    "-r", "24",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-ac", "2",
                    "-movflags", "+faststart",
                    "-threads", "2",
                    dst,
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=180)
                if proc.returncode != 0 or not _os.path.exists(dst) or _os.path.getsize(dst) < 1024:
                    err = proc.stderr.decode("utf-8", "ignore")[-500:]
                    raise HTTPException(500, f"ffmpeg normalize clip{i} failed: {err}")
                debug_log.append(f"norm{i}: {_os.path.getsize(dst)} bytes")
                normalized.append(dst)

            # NOTE: clips may now have different dimensions (we kept original aspect).
            # If they differ, concat-demuxer with -c copy will fail. Detect and either:
            #   - all same → fast stream-copy concat
            #   - different → use concat filter with re-scale (slower but correct)
            sizes = set()
            for n in normalized:
                probe = subprocess.run(
                    [ffbin, "-i", n], capture_output=True, timeout=30
                )
                err = probe.stderr.decode("utf-8", "ignore")
                m = re.search(r"(\d{2,5})x(\d{2,5})", err)
                if m:
                    sizes.add((int(m.group(1)), int(m.group(2))))
            uniform = len(sizes) <= 1
            debug_log.append(f"sizes={sizes} uniform={uniform}")

            out_path = _os.path.join(td, "final.mp4")
            if uniform:
                # 3a) Fast path: stream-copy concat
                list_path = _os.path.join(td, "files.txt")
                with open(list_path, "w", encoding="utf-8") as f:
                    for n in normalized:
                        safe = n.replace("\\", "/").replace("'", "'\\''")
                        f.write(f"file '{safe}'\n")
                cmd2 = [
                    ffbin, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                    "-c", "copy", "-movflags", "+faststart", out_path,
                ]
            else:
                # 3b) Mixed sizes → concat filter with rescale to first clip's size
                tw, th = next(iter(sizes))
                inputs: List[str] = []
                for n in normalized:
                    inputs.extend(["-i", n])
                # Build filter graph: scale each + concat
                fc_parts = []
                for idx in range(len(normalized)):
                    fc_parts.append(
                        f"[{idx}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{idx}];"
                    )
                concat_in = "".join(f"[v{idx}][{idx}:a]" for idx in range(len(normalized)))
                fc = "".join(fc_parts) + f"{concat_in}concat=n={len(normalized)}:v=1:a=1[v][a]"
                cmd2 = [
                    ffbin, "-y", *inputs,
                    "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "96k",
                    "-movflags", "+faststart",
                    "-threads", "2",
                    out_path,
                ]
            proc2 = subprocess.run(cmd2, capture_output=True, timeout=300)
            if proc2.returncode != 0 or not _os.path.exists(out_path):
                err = proc2.stderr.decode("utf-8", "ignore")[-500:]
                raise HTTPException(500, f"ffmpeg concat failed: {err}")

            final_size = _os.path.getsize(out_path)
            debug_log.append(f"final: {final_size} bytes")
            with open(out_path, "rb") as f:
                final_bytes = f.read()

        try:
            url = await _upload_to_tmpfiles(final_bytes, "manga_final.mp4", "video/mp4")
        except Exception as ue:
            raise HTTPException(500, f"upload to tmpfiles failed: {ue}")

        return {"url": url, "size": final_size, "clips": len(req.video_urls), "log": debug_log}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(500, f"concat error: {type(e).__name__}: {e} | log={debug_log}")


@app.get("/api/proxy-download")
async def proxy_download(url: str, filename: str = "download"):
    """Stream a remote file with Content-Disposition: attachment so the browser saves it.
    Bypasses CORS / cross-origin download attribute restrictions on third-party CDNs.
    """
    try:
        upstream = await http().get(
            url, follow_redirects=True, timeout=300,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if upstream.status_code != 200:
            raise HTTPException(400, f"upstream HTTP {upstream.status_code}")
        ct = upstream.headers.get("content-type", "application/octet-stream")
        # Sanitize filename
        safe_name = re.sub(r'[^\w.\-]+', '_', filename)[:120] or "download"
        return StreamingResponse(
            iter([upstream.content]),
            media_type=ct,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}"',
                "Content-Length": str(len(upstream.content)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"proxy error: {e}")


# ── IDEXX Neo Schedule Cache ──────────────────────────────────────────────────
_neo_schedule: dict = {}  # { "2026-04-27": [...appointments] }

class NeoAppointment(BaseModel):
    time: str
    patient: str
    owner: str
    room: str
    type: str
    notes: str
    provider: str = ""
    species: str = ""
    breed: str = ""
    sex: str = ""
    age: str = ""
    weight: str = ""
    color: str = ""
    dob: str = ""
    neutered: str = ""
    patient_id: str = ""

class NeoScheduleData(BaseModel):
    date: str
    appointments: List[NeoAppointment]

@app.post("/api/neo-schedule")
async def save_neo_schedule(data: NeoScheduleData):
    _neo_schedule[data.date] = [a.dict() for a in data.appointments]
    return {"ok": True, "date": data.date, "count": len(data.appointments)}

@app.get("/api/neo-schedule")
async def get_neo_schedule(date: str = "", refresh: bool = False):
    if not date:
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
            date = datetime.now(ZoneInfo("America/Vancouver")).strftime("%Y-%m-%d")
        except Exception:
            date = datetime.now().strftime("%Y-%m-%d")
    # On-demand fetch when nothing is cached for this date (or when forced)
    if refresh or date not in _neo_schedule:
        try:
            await scrape_neo_schedule(date_str=date)
        except Exception as e:
            print(f"[neo] on-demand scrape failed for {date}: {e}")
    return {"date": date, "appointments": _neo_schedule.get(date, [])}


@app.get("/api/balance")
async def balance():
    r = await http().get(
        f"{DATA999_BASE}/v1/skills/balance",
        headers={"Authorization": f"Bearer {DATA999_KEY}"},
        timeout=10,
    )
    return r.json()


# ── IDEXX Neo Scraper ─────────────────────────────────────────────────────────
def _age_from_dob(dob: str) -> str:
    """Convert YYYY-MM-DD birth date to '11 yrs 8 mos' style label."""
    try:
        from datetime import date as _date
        y, m, d = (int(x) for x in dob.split("-")[:3])
        b = _date(y, m, d); t = _date.today()
        months = (t.year - b.year) * 12 + (t.month - b.month) - (1 if t.day < b.day else 0)
        if months < 0: return ""
        years, mos = divmod(months, 12)
        if years and mos:  return f"{years} yrs {mos} mos"
        if years:          return f"{years} yrs"
        return f"{mos} mos"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────
#  Vet-intake-tool gate: verify a user-supplied IDEXX login by actually
#  POSTing to Neo. Returns a 30-day session token the frontend stores in
#  localStorage so users only re-auth once a month.
# ─────────────────────────────────────────────────────────────────────────
import hmac as _hmac, hashlib as _hashlib, base64 as _base64, time as _time
_AUTH_SECRET = os.environ.get("VET_AUTH_SECRET") or (NEO_PASS + "::vet-gate")

def _sign_token(username: str, expires_ts: int) -> str:
    msg = f"{username}|{expires_ts}".encode()
    sig = _hmac.new(_AUTH_SECRET.encode(), msg, _hashlib.sha256).digest()
    return _base64.urlsafe_b64encode(msg + b"::" + sig).decode().rstrip("=")

def _verify_token(token: str) -> bool:
    try:
        pad = "=" * (-len(token) % 4)
        raw = _base64.urlsafe_b64decode(token + pad)
        msg, sig = raw.rsplit(b"::", 1)
        expect = _hmac.new(_AUTH_SECRET.encode(), msg, _hashlib.sha256).digest()
        if not _hmac.compare_digest(sig, expect):
            return False
        _, exp = msg.decode().split("|")
        return int(exp) > int(_time.time())
    except Exception:
        return False

class VetLoginReq(BaseModel):
    username: str
    password: str

@app.post("/api/vet-verify-login")
async def vet_verify_login(req: VetLoginReq):
    """Verify by actually attempting a Neo login with user's credentials."""
    if not req.username or not req.password:
        return {"ok": False, "message": "用户名和密码必填"}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            login_page = await client.get(f"{NEO_BASE}/login/")
            lsoup = BeautifulSoup(login_page.text, "lxml")
            csrf_token = (lsoup.find("input", {"name": "csrf_neo_token"}) or {}).get("value", "")
            se_token   = (lsoup.find("input", {"name": "se_login_request"}) or {}).get("value", "")
            login = await client.post(f"{NEO_BASE}/login", data={
                "company_id": NEO_COMPANY, "username": req.username,
                "password": req.password, "submitted": "TRUE",
                "csrf_neo_token": csrf_token, "se_login_request": se_token,
            }, headers={"Referer": f"{NEO_BASE}/login/"})
            if "/login" in str(login.url):
                return {"ok": False, "message": "用户名或密码错误"}
        except Exception as e:
            return {"ok": False, "message": f"验证失败：{e}"}
    expires = int(_time.time()) + 30 * 86400
    return {"ok": True, "token": _sign_token(req.username, expires), "expires": expires}


async def _neo_login(client: httpx.AsyncClient) -> bool:
    """POST login to IDEXX Neo. Returns True on success."""
    if not NEO_USER or not NEO_PASS:
        return False
    login_page = await client.get(f"{NEO_BASE}/login/")
    lsoup = BeautifulSoup(login_page.text, "lxml")
    csrf_token = (lsoup.find("input", {"name": "csrf_neo_token"}) or {}).get("value", "")
    se_token   = (lsoup.find("input", {"name": "se_login_request"}) or {}).get("value", "")
    login = await client.post(f"{NEO_BASE}/login", data={
        "company_id": NEO_COMPANY, "username": NEO_USER,
        "password": NEO_PASS, "submitted": "TRUE",
        "csrf_neo_token": csrf_token, "se_login_request": se_token,
    }, headers={"Referer": f"{NEO_BASE}/login/"})
    return "/login" not in str(login.url)


class NeoAuthExpired(Exception):
    """Raised when a Neo response looks like the login page (session expired)."""
    pass

def _looks_like_login_page(body: str, final_url: str) -> bool:
    if "/login" in (final_url or ""):
        return True
    head = (body or "")[:3000].lower()
    # Login page contains a password input AND a username/csrf field; real history
    # responses contain neither.
    return ("type=\"password\"" in head or "type='password'" in head) and (
        "csrf_neo_token" in head or "se_login_request" in head or "name=\"username\"" in head
    )

async def _fetch_patient_history(client: httpx.AsyncClient, pid: str, days: int = 365) -> list[dict]:
    """Fetch /ajax/output/?page=modals/patient_history and parse consult entries."""
    from urllib.parse import quote
    from datetime import date as _date, timedelta as _td
    frm = (_date.today() - _td(days=days)).strftime("%d-%b %Y")
    url = (f"{NEO_BASE}/ajax/output/?page=modals/patient_history"
           f"&patient_id={pid}&from={quote(frm)}&to=&include_voided=false")
    r = await client.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
    if r.status_code != 200:
        return []
    if _looks_like_login_page(r.text, str(r.url)):
        raise NeoAuthExpired("history endpoint returned login page")
    soup = BeautifulSoup(r.text, "lxml")
    consults = []
    # Each consultation is in .consultation-list-item
    for item in soup.select(".consultation-list-item"):
        # Date + provider in the header
        header_txt = item.select_one(".consultation-list-item-header")
        header_str = header_txt.get_text(" ", strip=True) if header_txt else ""
        # Title / description
        title_el = item.select_one(".consultation-title, .consultation-list-item-description-row")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        # Vitals (weight etc)
        vitals_el = item.select_one(".consultation-list-item-vitals")
        vitals = vitals_el.get_text(" ", strip=True) if vitals_el else ""
        # Body — extract ONLY the clinical "Notes" section, drop billing/Qty table
        import re as _re
        raw = item.get_text("\n", strip=True)
        raw = _re.sub(r"\n{2,}", "\n", raw)
        # 1) cut from after the Notes header
        m = _re.search(r"(?:^|\n)Notes\n", raw)
        body = raw[m.end():] if m else raw
        # 2) cut at the start of the billing/products table
        bt = _re.search(r"\n(?:Qty|Product / Service|Total)\n", body)
        if bt:
            body = body[:bt.start()]
        # 3) drop residual billing-style lines
        keep = []
        for ln in body.split("\n"):
            s = ln.strip()
            if not s: continue
            if _re.match(r"^\$[\d,.]+$", s): continue                     # $124.64
            if s.startswith(("Batch No:", "Rx ID:", "(Includes ")): continue
            if s in ("Paid", "Unpaid", "View", "Staff", "Provider"): continue
            if _re.match(r"^#\d+$", s): continue                          # invoice no
            if _re.match(r"^\d{1,3}\.\d{2}$", s) and len(keep) > 0:        # bare price
                continue
            keep.append(s)
        body = "\n".join(keep).strip()
        # Capture consult_id (e.g. "#23041" in header)
        cid_m = _re.search(r"#(\d+)", header_str)
        consult_id = cid_m.group(1) if cid_m else ""
        consults.append({
            "header":     header_str[:200],
            "title":      title[:200],
            "vitals":     vitals[:120],
            "body":       body[:2500],
            "consult_id": consult_id,
        })
    return consults


async def _fetch_patient_rx(client: httpx.AsyncClient, pid: str) -> list[dict]:
    """Fetch /shared/prescriptions/list — returns simplified med list."""
    url = (f"{NEO_BASE}/shared/prescriptions/list"
           f"?patient_id={pid}&draw=1&start=0&length=50&include_voided=false")
    r = await client.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
    if r.status_code != 200:
        return []
    if _looks_like_login_page(r.text, str(r.url)):
        raise NeoAuthExpired("rx endpoint returned login page")
    try:
        rows = r.json()
    except Exception:
        # Non-JSON response on a normally-JSON endpoint also indicates auth issue
        if "<html" in (r.text or "").lower()[:200]:
            raise NeoAuthExpired("rx endpoint returned HTML instead of JSON")
        return []
    from datetime import date as _date, timedelta as _td
    cutoff = (_date.today() - _td(days=365)).isoformat()
    meds = []
    for row in rows:
        rx = row.get("prescription", row) if isinstance(row, dict) else {}
        if not rx or rx.get("voidedAt"): continue
        filled = (rx.get("filledAt") or "")[:10]
        # keep only fills within last 1 year (drop entries with no date too)
        if not filled or filled < cutoff:
            continue
        meds.append({
            "name":          rx.get("productName", ""),
            "rx_id":         str(rx.get("rxId", "")),
            "provider":      (rx.get("provider") or {}).get("name", ""),
            "instructions":  rx.get("instructions", ""),
            "quantity":      rx.get("quantity", ""),
            "refills":       rx.get("totalRefills", 0),
            "refills_left":  rx.get("refillsRemaining"),
            "filled_at":     filled,
        })
    # newest filled_at first
    meds.sort(key=lambda m: m["filled_at"], reverse=True)
    return meds


async def _fetch_patient_detail(client: httpx.AsyncClient, pid: str) -> dict:
    """Fetch /patients/view/{pid} and extract embedded JSON patient fields."""
    out = {"species": "", "breed": "", "sex": "", "weight": "",
           "color": "", "dob": "", "neutered": ""}
    try:
        r = await client.get(f"{NEO_BASE}/patients/view/{pid}")
        if r.status_code != 200:
            return out
        html = r.text
        import re
        def grab(key: str) -> str:
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', html)
            return m.group(1) if m else ""
        out["species"]  = grab("species") or grab("species_name")
        out["breed"]    = grab("breed")   or grab("breed_name")
        out["sex"]      = grab("sex")     or grab("gender_name")
        out["color"]    = grab("colour")  or grab("color")
        out["dob"]      = grab("date_of_birth")
        out["neutered"] = grab("neutered")
        # weight may be in a separate medical record block; try common JSON keys
        w = grab("weight") or grab("current_weight") or grab("last_weight")
        unit = grab("weight_unit") or grab("weight_units") or ""
        if w:
            out["weight"] = f"{w} {unit}".strip()
        else:
            # Fallback: scan rendered text for "X.XX kg" or "X.X lb" pattern
            soup = BeautifulSoup(html, "lxml")
            txt = soup.get_text(" ", strip=True)
            m = re.search(r"(\d{1,3}(?:\.\d{1,3})?)\s*(kg|lbs|lb)\b", txt, re.I)
            if m:
                out["weight"] = f"{m.group(1)} {m.group(2).lower()}"
        return out
    except Exception as e:
        print(f"[neo] patient {pid} fetch failed: {e}")
        return out


async def scrape_neo_schedule(date_str: str | None = None):
    """Login to IDEXX Neo and fetch a given date's schedule via the calendar API.

    If `date_str` is None, defaults to today (Vancouver local date).
    """
    if not NEO_USER or not NEO_PASS:
        print("[neo] NEO_USER/NEO_PASS not set, skipping")
        return
    # Use Vancouver local date (clinic timezone), not server UTC
    if not date_str:
        try:
            from zoneinfo import ZoneInfo
            date_str = datetime.now(ZoneInfo("America/Vancouver")).strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[neo] scraping {date_str}")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            if not await _neo_login(client):
                print("[neo] auth failed"); return

            # 3. Call calendar JSON API directly
            api_url = (f"{NEO_BASE}/appointments/getCalendarEventData"
                       f"?start={date_str}T00%3A00%3A00&end={date_str}T23%3A59%3A59")
            r = await client.get(api_url, headers={"X-Requested-With": "XMLHttpRequest"})
            data = r.json()
            events = data.get("events", [])
            print(f"[neo] API returned {len(events)} events")

            # 4. Build resource id → room name map
            room_map = {str(res["id"]): res["title"] for res in data.get("resources", [])}

            # 5. Parse events into appointments
            appts = []
            for i, ev in enumerate(events):
                if ev.get("is_block"):
                    continue
                title = ev.get("title", "")
                parts = title.split(";", 1)
                patient = parts[0].strip()
                owner   = parts[1].strip() if len(parts) > 1 else ""
                start   = ev.get("start", "")
                try:
                    from datetime import datetime as _dt
                    t = _dt.strptime(start, "%Y-%m-%d %H:%M:%S")
                    time_fmt = t.strftime("%-I:%M%p").lower()
                except Exception:
                    time_fmt = start[11:16]
                notes = (ev.get("reason") or ev.get("popup_title_text") or "")
                # Strip notes that are just the provider/doctor name (no real chief complaint).
                # e.g. "Chang-Chi Chiu, DVM" or "Dr. Rex Yang" leaking into the reason field.
                _notes_clean = (notes or "").strip()
                _provider = (ev.get("provider") or "").strip()
                if _notes_clean and (
                    _notes_clean == _provider
                    or re.match(r"^(Dr\.?\s+[A-Z][\w\-]+(\s+[A-Z][\w\-]+)*|[A-Z][\w\-]+(\s+[A-Z][\w\-]+){0,3},?\s*DVM)\.?$", _notes_clean)
                ):
                    notes = ""
                pid = (ev.get("patient_id") or ev.get("patientId")
                       or ev.get("patient") or ev.get("pet_id") or "")
                appts.append({
                    "time":       time_fmt,
                    "patient":    patient,
                    "owner":      owner,
                    "room":       room_map.get(str(ev.get("resourceId", "")), ""),
                    "type":       ev.get("type_description", ""),
                    "notes":      notes,
                    "provider":   ev.get("provider", ""),
                    "species":    "",
                    "breed":      "",
                    "sex":        "",
                    "age":        "",
                    "weight":     "",
                    "color":      "",
                    "dob":        "",
                    "neutered":   "",
                    "patient_id": str(pid),
                })

            # 6. Enrich each appointment with patient details from /patients/view/{id}
            seen: dict = {}
            for a in appts:
                pid = a["patient_id"]
                if not pid:
                    continue
                if pid not in seen:
                    seen[pid] = await _fetch_patient_detail(client, pid)
                a.update(seen[pid])
                # Compute age from dob
                if a.get("dob") and not a.get("age"):
                    a["age"] = _age_from_dob(a["dob"])

            appts.sort(key=lambda a: a["time"])
            _neo_schedule[date_str] = appts
            print(f"[neo] cached {len(appts)} appointments for {date_str}")
    except Exception as e:
        print(f"[neo] error: {e}")


# ── Scheduler: 9am–10pm every hour ────────────────────────────────────────────
_scheduler = AsyncIOScheduler(timezone="America/Vancouver")

@app.on_event("startup")
async def start_scheduler():
    _scheduler.add_job(
        scrape_neo_schedule,
        CronTrigger(hour="9-22", minute="0", timezone="America/Vancouver"),
    )
    _scheduler.start()
    print("[scheduler] started — IDEXX Neo scrape every hour 9am–10pm PT")
    # Always run once on startup to populate cache immediately
    asyncio.create_task(scrape_neo_schedule())


@app.post("/api/neo-schedule/refresh")
async def refresh_neo_schedule():
    """Manually trigger a schedule scrape."""
    asyncio.create_task(scrape_neo_schedule())
    return {"ok": True, "message": "scrape triggered"}


_neo_history_cache: dict = {}  # { pid: {"at": ts, "consults": [...]} }

# Persistent logged-in Neo session — reused across requests so we skip
# the (slow) login handshake every time. Re-login is done lazily if a
# request returns 401/redirects to /login.
_neo_session: dict = {"client": None, "login_lock": None}

async def _get_neo_session() -> httpx.AsyncClient:
    """Return a logged-in httpx client, logging in once and reusing it."""
    import asyncio as _asyncio
    if _neo_session["login_lock"] is None:
        _neo_session["login_lock"] = _asyncio.Lock()
    async with _neo_session["login_lock"]:
        c = _neo_session["client"]
        if c is None:
            c = httpx.AsyncClient(follow_redirects=True, timeout=20)
            if not await _neo_login(c):
                await c.aclose()
                _neo_session["client"] = None
                raise HTTPException(503, "neo auth failed")
            _neo_session["client"] = c
        return c

async def _ensure_neo_authed(client: httpx.AsyncClient) -> bool:
    """If a probe shows the session has expired, re-login in place."""
    probe = await client.get(f"{NEO_BASE}/", timeout=10)
    if "/login" in str(probe.url):
        return await _neo_login(client)
    return True

@app.get("/api/neo-history")
async def get_neo_history(pid: str, days: int = 365, refresh: bool = False):
    """Fetch and cache a patient's clinical history + active medications.
    Cached for 30 min per pid (refresh=true bypasses cache)."""
    import time, asyncio as _asyncio
    now = time.time()
    cached = _neo_history_cache.get(pid)
    if cached and not refresh and (now - cached["at"]) < 1800:
        return {"pid": pid, "cached": True,
                "consults": cached["consults"],
                "medications": cached.get("medications", [])}
    try:
        client = await _get_neo_session()
        # Run history + rx in PARALLEL (was sequential — biggest speedup)
        try:
            consults, meds = await _asyncio.gather(
                _fetch_patient_history(client, pid, days),
                _fetch_patient_rx(client, pid),
            )
        except Exception:
            # Session may have expired — re-login once and retry
            if await _neo_login(client):
                consults, meds = await _asyncio.gather(
                    _fetch_patient_history(client, pid, days),
                    _fetch_patient_rx(client, pid),
                )
            else:
                # Drop stale session so next call gets a fresh one
                _neo_session["client"] = None
                raise
        # Don't cache fully-empty results — they almost always indicate a transient
        # auth/parse failure and we don't want to lock the user out for 30 min.
        if consults or meds:
            _neo_history_cache[pid] = {"at": now, "consults": consults, "medications": meds}
        return {"pid": pid, "cached": False,
                "consults": consults, "medications": meds}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"history fetch failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
#  Patient files (Documents tab) — list + AI summary of attached files
# ─────────────────────────────────────────────────────────────────────────
_neo_file_summary_cache: dict = {}  # { file_id: {"at": ts, "summary": str} }

@app.get("/api/neo-patient-files")
async def neo_patient_files(pid: str):
    """List patient's attached files (Documents tab in Neo).
    Returns lightweight metadata only — actual download happens server-side
    in /api/neo-file-summary so we don't expose CDN-signed URLs to the browser.
    """
    try:
        client = await _get_neo_session()
        url = f"{NEO_BASE}/files/patient/{pid}"
        r = await client.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
        if r.status_code != 200 or _looks_like_login_page(r.text, str(r.url)):
            # Try once more after re-login
            if await _neo_login(client):
                r = await client.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
            if r.status_code != 200:
                raise HTTPException(502, f"neo files {r.status_code}")
        try:
            data = r.json()
        except Exception:
            raise HTTPException(502, "neo files: non-JSON response")
        files = data.get("files", []) or []
        out = []
        for f in files:
            created = ((f.get("createdAtLocalBranchTime") or {}).get("date") or "")[:10]
            out.append({
                "id":         f.get("id"),
                "filename":   f.get("filename") or "",
                "title":      f.get("title") or "",
                "category":   f.get("category") or "",
                "mime":       f.get("mimeType") or "",
                "size":       f.get("fileSize") or 0,
                "created":    created,
                "type":       f.get("type") or "",
            })
        # Newest first
        out.sort(key=lambda x: x.get("created", ""), reverse=True)
        return {"pid": pid, "files": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"file list failed: {e}")


async def _fetch_file_bytes(client: httpx.AsyncClient, pid: str, file_id: int) -> tuple[bytes, str, str]:
    """Look up the cdnLink for one file_id and download the bytes.
    Returns (bytes, filename, mime). Re-logs in once if needed."""
    list_url = f"{NEO_BASE}/files/patient/{pid}"
    r = await client.get(list_url, headers={"X-Requested-With": "XMLHttpRequest"})
    if r.status_code != 200 or _looks_like_login_page(r.text, str(r.url)):
        if await _neo_login(client):
            r = await client.get(list_url, headers={"X-Requested-With": "XMLHttpRequest"})
    data = r.json()
    target = next((f for f in (data.get("files") or []) if int(f.get("id") or 0) == int(file_id)), None)
    if not target:
        raise HTTPException(404, f"file {file_id} not found on patient {pid}")
    cdn = target.get("cdnLink") or ""
    if not cdn:
        raise HTTPException(502, "neo file has no cdnLink")
    # Download — same authed httpx client (Neo CDN paths require the session cookie)
    dl = await client.get(cdn, follow_redirects=True, timeout=60)
    if dl.status_code != 200:
        raise HTTPException(502, f"file download {dl.status_code}")
    return dl.content, (target.get("filename") or "file.bin"), (target.get("mimeType") or "application/octet-stream")


def _extract_text_from_pdf(data: bytes) -> str:
    """Best-effort PDF text extraction. Tries pypdf, falls back to a no-op."""
    try:
        from pypdf import PdfReader  # type: ignore
        import io as _io
        rd = PdfReader(_io.BytesIO(data))
        chunks = []
        for p in rd.pages[:60]:  # cap pages so a 200-page record doesn't blow up
            try:
                chunks.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def _extract_text_from_image(data: bytes, mime: str) -> str:
    """OCR an image attachment. Returns "" if Tesseract isn't available."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        import io as _io
        img = Image.open(_io.BytesIO(data))
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


def _pdf_to_images(data: bytes, max_pages: int = 8) -> list[bytes]:
    """Render PDF pages to JPEG bytes (for vision OCR). Uses PyMuPDF (fitz) if available."""
    try:
        import fitz  # type: ignore (PyMuPDF)
        doc = fitz.open(stream=data, filetype="pdf")
        out = []
        for i, page in enumerate(doc):
            if i >= max_pages: break
            pm = page.get_pixmap(dpi=150)
            out.append(pm.tobytes("jpeg"))
        doc.close()
        return out
    except Exception:
        return []


async def _vision_ocr_image(data: bytes, mime: str = "image/jpeg") -> str:
    """OCR via GPT vision (works without any system tesseract install)."""
    import base64 as _b64, json as _json
    b64 = _b64.b64encode(data).decode()
    data_url = f"data:{mime};base64,{b64}"
    try:
        async with httpx.AsyncClient(timeout=90) as ai:
            r = await ai.post(
                "https://api.ai6700.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract ALL visible text from this image, verbatim. Preserve line breaks, dates, dosages, numbers exactly. No summary — just the raw text."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                },
            )
            if r.status_code != 200:
                return ""
            j = r.json()
            return (((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""


@app.get("/api/neo-file-summary")
async def neo_file_summary(pid: str, file_id: int, refresh: bool = False):
    """Download one of a patient's attached files server-side, extract text,
    and have GPT summarize history / meds / vaccines.
    Cached 24h per file_id (refresh=true bypasses cache)."""
    import time as _time, json as _json
    cache_key = str(file_id)
    cached = _neo_file_summary_cache.get(cache_key)
    if cached and not refresh and (_time.time() - cached["at"]) < 604800:  # 7 days
        return {"file_id": file_id, "cached": True, **cached["payload"]}

    try:
        client = await _get_neo_session()
        try:
            data, filename, mime = await _fetch_file_bytes(client, pid, file_id)
        except HTTPException:
            raise
        except Exception:
            # Try one more time after re-login
            if await _neo_login(client):
                data, filename, mime = await _fetch_file_bytes(client, pid, file_id)
            else:
                _neo_session["client"] = None
                raise

        # Extract text by mime type
        text = ""
        kind = "unknown"
        m = (mime or "").lower()
        if "pdf" in m:
            kind = "pdf"
            text = _extract_text_from_pdf(data)
            # Fallback 1: render pages locally if PyMuPDF is available
            if not text.strip():
                pages = _pdf_to_images(data, max_pages=8)
                if pages:
                    kind = "pdf-ocr"
                    chunks = []
                    for i, jpg in enumerate(pages):
                        ocr = await _vision_ocr_image(jpg, "image/jpeg")
                        if ocr:
                            chunks.append(f"--- page {i+1} ---\n{ocr}")
                    text = "\n\n".join(chunks)
            # Fallback 2: try sending the PDF bytes directly as a vision input
            # (some OpenAI-compatible endpoints accept application/pdf in image_url)
            if not text.strip():
                ocr = await _vision_ocr_image(data, "application/pdf")
                if ocr.strip():
                    kind = "pdf-direct"
                    text = ocr
        elif m.startswith("image/"):
            kind = "image"
            text = _extract_text_from_image(data, mime)
            if not text.strip():
                # Fallback to vision OCR
                kind = "image-ocr"
                text = await _vision_ocr_image(data, mime)
        elif "text" in m:
            kind = "text"
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        text = (text or "").strip()
        if not text:
            payload = {
                "filename": filename, "mime": mime, "kind": kind,
                "size": len(data),
                "summary": "（无法从此文件中提取文字。可能是扫描件未做 OCR / 图片质量太低 / 加密 PDF。）",
                "raw_chars": 0,
            }
            _neo_file_summary_cache[cache_key] = {"at": _time.time(), "payload": payload}
            return {"file_id": file_id, "cached": False, **payload}

        # Trim to a sane size for the LLM
        snippet = text if len(text) <= 18000 else text[:18000] + "\n...(truncated)"

        sys_prompt = (
            "你是兽医技师助手。下面是一份从外院调来的病历文件原文（可能是 PDF/图片 OCR 出来的纯文本，"
            "格式可能比较乱）。请结构化总结，让接诊技师 30 秒内能掌握重点。\n\n"
            "输出固定 4 个栏目（即使该栏没有内容也要保留并写「无」）：\n"
            "【📋 概况】1–2 句话：动物名/品种/性别/年龄；这份文件大致是什么（例如：猫白血病检查报告 / 转诊信 / 既往就诊摘要）。\n"
            "【🩺 重要病史 (HISTORY)】列出最相关的诊断、慢性病、手术、住院记录。日期都保留。\n"
            "【💊 用药 (MEDS)】列所有提到的药名 + 剂量 + 频率 + 起止日期（拿不到日期就写「日期未提」）。\n"
            "【💉 疫苗 (VACCINES)】列疫苗名、日期、下次到期。\n\n"
            "规则：\n"
            "- 全部用中文输出，专有词（药名/疫苗品牌/检查项目）保留英文原文。\n"
            "- 不要瞎编。原文没写的就写「无」或「未提」。\n"
            "- 列表用 `- ` 开头，简洁。"
        )

        async with httpx.AsyncClient(timeout=90) as ai:
            ar = await ai.post(
                "https://api.ai6700.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": f"文件名：{filename}\n类型：{kind}\n\n--- 原文 ---\n{snippet}"},
                    ],
                },
            )
            if ar.status_code != 200:
                raise HTTPException(502, f"AI {ar.status_code}: {ar.text[:200]}")
            jd = ar.json()
            summary = (((jd.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

        payload = {
            "filename": filename, "mime": mime, "kind": kind,
            "size": len(data), "raw_chars": len(text),
            "summary": summary or "（AI 未返回内容）",
        }
        _neo_file_summary_cache[cache_key] = {"at": _time.time(), "payload": payload}
        return {"file_id": file_id, "cached": False, **payload}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"file summary failed: {e}")


class VisionSummaryReq(BaseModel):
    images_b64: list[str]      # JPEG/PNG base64 strings (no data: prefix)
    filename: str = ""
    mime_hint: str = "image/jpeg"
    file_id: int = 0           # optional — if set, result is cached under this id

@app.post("/api/neo-vision-summary")
async def neo_vision_summary(req: VisionSummaryReq):
    """Browser sends PDF pages already rasterized to base64 images.
    We OCR each page via vision API and feed the joined text into the same
    summarizer used by /api/neo-file-summary."""
    if not req.images_b64:
        raise HTTPException(400, "no images")
    if len(req.images_b64) > 12:
        raise HTTPException(400, "too many pages (max 12)")

    # If we already cached this file_id, return the cached payload immediately
    import time as _time
    if req.file_id:
        ck = str(req.file_id)
        cached = _neo_file_summary_cache.get(ck)
        if cached and (_time.time() - cached["at"]) < 604800:
            p = cached["payload"]
            return {"filename": p.get("filename", req.filename),
                    "summary": p.get("summary", ""),
                    "raw_chars": p.get("raw_chars", 0),
                    "pages": len(req.images_b64),
                    "cached": True}

    chunks = []
    for i, b64 in enumerate(req.images_b64):
        try:
            import base64 as _b64
            raw = _b64.b64decode(b64)
        except Exception:
            continue
        ocr = await _vision_ocr_image(raw, req.mime_hint or "image/jpeg")
        if ocr.strip():
            chunks.append(f"--- page {i+1} ---\n{ocr}")
    text = "\n\n".join(chunks).strip()
    if not text:
        return {"filename": req.filename, "summary": "（页面 OCR 没识别出任何文字。）", "raw_chars": 0, "pages": len(req.images_b64)}

    snippet = text if len(text) <= 18000 else text[:18000] + "\n...(truncated)"
    sys_prompt = (
        "你是兽医技师助手。下面是一份从外院调来的病历文件原文（PDF 经 OCR 转出的纯文本，"
        "格式可能比较乱）。请结构化总结，让接诊技师 30 秒内能掌握重点。\n\n"
        "输出固定 4 个栏目（即使该栏没有内容也要保留并写「无」）：\n"
        "【📋 概况】1–2 句话：动物名/品种/性别/年龄；这份文件大致是什么。\n"
        "【🩺 重要病史 (HISTORY)】列出最相关的诊断、慢性病、手术、住院记录。日期都保留。\n"
        "【💊 用药 (MEDS)】列所有提到的药名 + 剂量 + 频率 + 起止日期（拿不到日期就写「日期未提」）。\n"
        "【💉 疫苗 (VACCINES)】列疫苗名、日期、下次到期。\n\n"
        "规则：全部用中文输出，专有词保留英文原文。原文没写的就写「无」或「未提」。列表用 `- ` 开头。"
    )
    async with httpx.AsyncClient(timeout=90) as ai:
        ar = await ai.post(
            "https://api.ai6700.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-5.4-mini",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": f"文件名：{req.filename}\n页数：{len(req.images_b64)}\n\n--- OCR 原文 ---\n{snippet}"},
                ],
            },
        )
        if ar.status_code != 200:
            raise HTTPException(502, f"AI {ar.status_code}: {ar.text[:200]}")
        jd = ar.json()
        summary = (((jd.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    payload = {
        "filename": req.filename,
        "summary": summary or "（AI 未返回内容）",
        "raw_chars": len(text),
        "pages": len(req.images_b64),
        "kind": "pdf-ocr-browser",
        "size": 0,
        "mime": "application/pdf",
    }
    # Persist so the next click within 7 days returns instantly via /api/neo-file-summary
    if req.file_id:
        _neo_file_summary_cache[str(req.file_id)] = {"at": _time.time(), "payload": payload}
    return payload


@app.get("/api/neo-file-bytes")
async def neo_file_bytes(pid: str, file_id: int):
    """Stream file bytes (e.g. PDF) back to the browser so it can render pages with PDF.js."""
    try:
        client = await _get_neo_session()
        try:
            data, filename, mime = await _fetch_file_bytes(client, pid, file_id)
        except Exception:
            if await _neo_login(client):
                data, filename, mime = await _fetch_file_bytes(client, pid, file_id)
            else:
                _neo_session["client"] = None
                raise
        from fastapi.responses import Response
        return Response(content=data, media_type=mime or "application/octet-stream",
                        headers={"Content-Disposition": f'inline; filename="{filename}"',
                                 "Access-Control-Allow-Origin": "*"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"file bytes failed: {e}")


@app.get("/api/neo-files-probe")
async def neo_files_probe(pid: str):
    """Probe several likely Neo paths to find where patient attachments live."""
    candidates = [
        # ajax/output style
        f"{NEO_BASE}/ajax/output/?page=modals/patient_files&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=modals/patient_documents&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=modals/patient_attachments&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=modals/files&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=modals/documents&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=patients/files&patient_id={pid}",
        f"{NEO_BASE}/ajax/output/?page=patients/files_list&patient_id={pid}",
        # tab style
        f"{NEO_BASE}/patients/{pid}/files-tab",
        f"{NEO_BASE}/patients/{pid}/tabs/files",
        # shared list style (matches /shared/prescriptions/list pattern)
        f"{NEO_BASE}/shared/files/list?patient_id={pid}&draw=1&start=0&length=50",
        f"{NEO_BASE}/shared/patient_files/list?patient_id={pid}&draw=1&start=0&length=50",
        f"{NEO_BASE}/shared/documents/list?patient_id={pid}&draw=1&start=0&length=50",
        f"{NEO_BASE}/shared/attachments/list?patient_id={pid}&draw=1&start=0&length=50",
        # page-data style
        f"{NEO_BASE}/patients/{pid}/page-data",
        # patient main page (HTML — search for file/document references)
        f"{NEO_BASE}/patients/{pid}",
    ]
    out = []
    try:
        client = await _get_neo_session()
        for u in candidates:
            try:
                r = await client.get(u, headers={"X-Requested-With": "XMLHttpRequest"}, timeout=15)
                body = r.text or ""
                snip = body[:400]
                ct = r.headers.get("content-type", "")
                # If body is HTML (large), also extract any href/src that mentions
                # files/documents/attachments/upload/download — these are signals
                # of where the file UI lives.
                hints = []
                if "html" in ct.lower() and len(body) > 1000:
                    import re as _re
                    for m in _re.finditer(r'(?:href|src|data-url|data-href|action)="([^"]{6,200})"', body):
                        v = m.group(1).lower()
                        if any(k in v for k in ("file","document","attach","upload","download")):
                            if v not in hints:
                                hints.append(m.group(1))
                                if len(hints) >= 30: break
                out.append({
                    "url": u,
                    "status": r.status_code,
                    "ct": ct,
                    "len": len(body),
                    "login": _looks_like_login_page(body, str(r.url)),
                    "snip": snip,
                    "hints": hints,
                })
            except Exception as e:
                out.append({"url": u, "error": str(e)})
        return {"pid": pid, "results": out}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/neo-history-debug")
async def neo_history_debug(pid: str, days: int = 365):
    """Diagnostic: dumps raw URL, status, redirect target, HTML head, and selector counts."""
    from urllib.parse import quote
    from datetime import date as _date, timedelta as _td
    frm = (_date.today() - _td(days=days)).strftime("%d-%b %Y")
    url = (f"{NEO_BASE}/ajax/output/?page=modals/patient_history"
           f"&patient_id={pid}&from={quote(frm)}&to=&include_voided=false")
    try:
        client = await _get_neo_session()
        r = await client.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
        body = r.text or ""
        soup = BeautifulSoup(body, "lxml")
        sels = {
            ".consultation-list-item": len(soup.select(".consultation-list-item")),
            "[class*='consultation']": len(soup.select("[class*='consultation']")),
            ".timeline-item": len(soup.select(".timeline-item")),
            "tr[data-consultation-id]": len(soup.select("tr[data-consultation-id]")),
        }
        # Try to autodetect class names that contain 'consult'
        classes = set()
        for el in soup.find_all(class_=True)[:5000]:
            for c in el.get("class", []):
                if "consult" in c.lower() or "history" in c.lower() or "timeline" in c.lower():
                    classes.add(c)
        return {
            "url": url,
            "status": r.status_code,
            "final_url": str(r.url),
            "looks_like_login": "/login" in str(r.url) or "<form" in body[:2000].lower() and "password" in body[:2000].lower(),
            "len": len(body),
            "selector_counts": sels,
            "candidate_classes": sorted(classes)[:50],
            "head": body[:1500],
        }
    except Exception as e:
        return {"error": str(e), "url": url}


class ConsultUpdateS(BaseModel):
    pid: str = ""
    consult_id: str
    original_s: str = ""
    corrected_s: str = ""
    corrections: list[dict] = []   # [{original, corrected, context}]

@app.post("/api/neo-consult-update-s")
async def neo_consult_update_s(data: ConsultUpdateS):
    """Update only the S (Subjective) section of a consult's notes.

    Flow:
      1. GET  /consultations/{cid}/notes  → {notes: HTML, notesVersion: "..."}
      2. Parse HTML, locate the <td> that follows <strong>S = Subjective Information</strong>
      3. Apply each {original→corrected} word-boundary replacement INSIDE that td only
      4. PUT /consultations/{cid}/notes with {notes: modified_html, notesVersion: same}
    """
    if not data.consult_id:
        raise HTTPException(400, "consult_id required")
    if not data.corrections:
        return {"ok": False, "message": "no corrections to apply"}
    cid = data.consult_id
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            if not await _neo_login(client):
                raise HTTPException(503, "neo auth failed")

            # XSRF token is set by Neo as a cookie at login
            xsrf = (client.cookies.get("XSRF-TOKEN")
                    or client.cookies.get("csrf_neo_cookie") or "")
            get_api = f"{NEO_BASE}/consultations/{cid}/page-data"
            put_api = f"{NEO_BASE}/consultations/{cid}/notes"
            base_headers = {
                "X-XSRF-TOKEN":       xsrf,
                "X-Neo-Core-Version": "126.13.1",
                "Accept":             "application/json, text/plain, */*",
                "Referer":            f"{NEO_BASE}/consultations/view/{cid}",
                "Origin":             NEO_BASE,
                "X-Requested-With":   "XMLHttpRequest",
            }

            # 1) Fetch full consult page-data (notes + version live somewhere inside)
            r = await client.get(get_api, headers=base_headers)
            if r.status_code != 200:
                return {"ok": False, "message": f"GET page-data failed: {r.status_code}",
                        "diagnostic": {"resp": r.text[:400]}}
            try:
                payload = r.json()
            except Exception:
                return {"ok": False, "message": "GET page-data returned non-JSON",
                        "diagnostic": {"resp": r.text[:400]}}

            # Deep-walk JSON to find the notes HTML + matching version
            def _deep_find(obj, key):
                if isinstance(obj, dict):
                    if key in obj:
                        return obj[key]
                    for v in obj.values():
                        rr = _deep_find(v, key)
                        if rr is not None:
                            return rr
                elif isinstance(obj, list):
                    for v in obj:
                        rr = _deep_find(v, key)
                        if rr is not None:
                            return rr
                return None

            current_html = _deep_find(payload, "notes") or ""
            # Neo's optimistic-concurrency token is `notesUpdatedAtLocal`
            # (a "YYYY-MM-DD HH:MM:SS" string), NOT `notesVersion`.
            version = (_deep_find(payload, "notesUpdatedAtLocal")
                       or _deep_find(payload, "notesVersion")
                       or "")
            if not current_html:
                return {"ok": False, "message": "no notes field found in page-data",
                        "diagnostic": {"top_keys": list(payload.keys()) if isinstance(payload, dict) else "non-dict"}}

            # 2-3) Locate the S section in the RAW HTML string and patch it in
            #      place — do NOT round-trip through BeautifulSoup.str(soup),
            #      which destroys Neo's whitespace/paragraph layout.
            import re as _re
            def _patch_s_in_raw_html(raw_html, corrections_list):
                # Find "S = Subjective Information" anywhere in the raw HTML
                m_s = _re.search(r"S\s*=\s*Subjective\s+Information", raw_html, _re.IGNORECASE)
                if not m_s:
                    return None, None, "could not locate S = Subjective marker"
                # Find the next "X = " section header AFTER the S marker
                # (O = Objective / A = Assessment / P = Plan, etc.)
                m_next = _re.search(
                    r"[OAP]\s*=\s*(Objective|Assessment|Plan)",
                    raw_html[m_s.end():],
                    _re.IGNORECASE,
                )
                # The S "zone" is everything between the S marker and the next section
                # header. We do replacements inside this zone only.
                if m_next:
                    zone_start = m_s.end()
                    zone_end   = m_s.end() + m_next.start()
                else:
                    zone_start = m_s.end()
                    zone_end   = len(raw_html)
                zone = raw_html[zone_start:zone_end]
                applied_local = []
                # Split the zone into [tag, text, tag, text, ...] segments so
                # we only ever replace inside TEXT segments — HTML tags
                # (everything between < and >) are kept literally untouched.
                # This guarantees we cannot break style="width:..." attributes
                # or tag names like <p>, <td>, <tr>, etc.
                parts = _re.split(r"(<[^>]+>)", zone)  # odd indices = tags
                # Helper: normalize HTML text for fuzzy matching against AI-supplied
                # plain-text sentences. Decodes &amp;/&nbsp;/&lt;/&gt;/&quot;,
                # collapses any whitespace run (incl. NBSP) to a single space.
                import html as _html
                def _norm(s):
                    s = _html.unescape(s or "")
                    s = s.replace("\u00a0", " ")
                    s = _re.sub(r"\s+", " ", s)
                    return s.strip()

                for c in corrections_list:
                    o = (c.get("original") or "").strip()
                    cc = (c.get("corrected") or "").strip()
                    if not o or not cc or o == cc:
                        continue
                    o_norm = _norm(o)
                    if not o_norm:
                        continue
                    # Build a tolerant pattern: any whitespace run in `original`
                    # matches any whitespace run in HTML (incl. &nbsp;/NBSP).
                    # Word boundaries dropped — sentence-level corrections often
                    # end with punctuation, where \b doesn't fire.
                    tokens = o_norm.split(" ")
                    pat = _re.compile(
                        r"(?:\s|&nbsp;|\u00a0)*".join(_re.escape(t) for t in tokens),
                        _re.IGNORECASE,
                    )
                    total = 0
                    # Pass 1: replace inside each text segment (fast path — most
                    # AI-suggested sentences live entirely within one text node).
                    for i in range(0, len(parts), 2):
                        parts[i], n = pat.subn(cc, parts[i])
                        total += n
                    # Pass 2 (fallback): if no match in any text segment, the
                    # sentence may be split across <br>/<p> tags. Try matching
                    # across the whole zone, allowing tags between tokens.
                    if total == 0:
                        cross_pat = _re.compile(
                            r"(?:\s|&nbsp;|\u00a0|<[^>]+>)*".join(_re.escape(t) for t in tokens),
                            _re.IGNORECASE,
                        )
                        new_zone_full, n2 = cross_pat.subn(cc, "".join(parts))
                        if n2:
                            parts = [new_zone_full]  # collapse — we lose tag granularity here, but that's acceptable
                            total = n2
                    applied_local.append({"original": o, "corrected": cc, "count": total})
                new_zone = "".join(parts)
                if not any(a["count"] for a in applied_local):
                    # Surface a useful diagnostic so the user can see what actually
                    # lives in the S zone (first 400 chars of normalized text).
                    sample = _norm(_re.sub(r"<[^>]+>", " ", zone))[:400]
                    return None, applied_local, "none of the corrections matched the S text. S zone preview: " + sample
                patched = raw_html[:zone_start] + new_zone + raw_html[zone_end:]
                return patched, applied_local, None

            new_full_html, applied, err = _patch_s_in_raw_html(current_html, data.corrections)
            if err:
                return {"ok": False, "message": err,
                        "diagnostic": {"applied": applied or []}}

            # 4) PUT back — with auto-retry on stale-version conflict.
            #    On conflict Neo returns 200/4xx with body
            #      {"message":"Object is already modified",
            #       "consultationNotes":{"notes":"...","notesVersion":"..."}}
            #    We re-apply the same corrections to the freshly-returned notes
            #    HTML and PUT once more.
            async def _do_put(html_to_save, ver):
                # Send BOTH possible field names — harmless if one is ignored.
                return await client.put(
                    put_api,
                    headers={**base_headers, "Content-Type": "application/json"},
                    json={"notes": html_to_save,
                          "notesUpdatedAtLocal": ver,
                          "notesVersion": ver},
                )

            def _apply_corrections_to_full_html(full_html):
                # Reuse the in-place raw-string patcher to preserve formatting
                return _patch_s_in_raw_html(full_html, data.corrections)

            # Helper: detect "Object is already modified" regardless of status code.
            # Robustly locates fresh notes + version even if they're nested
            # somewhere deep, or under a slightly different name.
            import time as _time
            def _conflict_payload(resp):
                if "Object is already modified" not in (resp.text or ""):
                    return None
                try:
                    j = resp.json()
                except Exception:
                    return None
                if not isinstance(j, dict): return None
                # 1) try the documented shape first
                cn = j.get("consultationNotes") or {}
                fh = cn.get("notes") or _deep_find(j, "notes") or ""
                fv = (cn.get("notesUpdatedAtLocal")
                      or _deep_find(j, "notesUpdatedAtLocal")
                      or cn.get("notesVersion")
                      or _deep_find(j, "notesVersion")
                      or "")
                if not fh:
                    return None
                if not fv:
                    # last resort: fresh timestamp (Neo uses ms-epoch-like ints)
                    fv = str(int(_time.time() * 1000))
                return {"notes": fh, "notesVersion": fv}

            # Try up to 4 times: each time a conflict comes back, re-apply
            # corrections to the latest notes Neo gave us and PUT again.
            current_html_full = new_full_html
            current_version   = version
            current_applied   = applied
            last_resp         = None
            for attempt in range(4):
                put = await _do_put(current_html_full, current_version)
                last_resp = put
                conflict = _conflict_payload(put)
                if conflict is None:
                    if put.status_code == 200:
                        return {"ok": True, "applied": current_applied,
                                "message": f"saved {sum(a['count'] for a in current_applied)} edits" + (f" (retry {attempt})" if attempt else "")}
                    return {"ok": False, "message": f"PUT failed: {put.status_code}",
                            "diagnostic": {"resp": put.text[:400], "applied": current_applied}}
                # rebuild from fresh notes returned by server
                retry_html, retry_applied, err = _apply_corrections_to_full_html(conflict["notes"])
                if err:
                    return {"ok": False, "message": f"retry parse failed: {err}",
                            "diagnostic": {"applied": current_applied}}
                current_html_full = retry_html
                current_version   = conflict["notesVersion"]
                current_applied   = retry_applied
            # Dump the FULL last response + a parsed-keys map so we can see
            # what version field Neo actually returns.
            dbg = {"applied": current_applied}
            if last_resp is not None:
                dbg["status"] = last_resp.status_code
                dbg["full_resp"] = last_resp.text
                try:
                    jj = last_resp.json()
                    def _walk(o, path=""):
                        out = []
                        if isinstance(o, dict):
                            for k, v in o.items():
                                p = f"{path}.{k}" if path else k
                                if isinstance(v, (dict, list)):
                                    out.extend(_walk(v, p))
                                else:
                                    s = str(v)
                                    out.append(f"{p}={s[:80]}")
                        elif isinstance(o, list):
                            for i, v in enumerate(o[:3]):
                                out.extend(_walk(v, f"{path}[{i}]"))
                        return out
                    dbg["fields"] = _walk(jj)
                    dbg["last_version_sent"] = current_version
                except Exception as e:
                    dbg["json_err"] = str(e)
            return {"ok": False, "message": "still conflicting after 4 retries",
                    "diagnostic": dbg}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "message": f"exception: {e}"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
