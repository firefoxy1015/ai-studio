import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AI Studio")

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
LINGKEAI_S6 = "Chengchen@630"
LINGKEAI_MODEL_IDS = {
    "grok-4.2": 94,
    "grok-4.2-image": 48,
    "grok-video-3": 31,
    "grok-video-3-plus": 63,
    "claude-opus-4-7": 90,
    "claude-sonnet-4-6": 38,
    "claude-opus-4-6": 33,
    "gemini-3.1-pro-preview": 39,
    "gemini-3-pro-preview": 5,
    "doubao-seed-2-0-pro-260215": 40,
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


class GenerateRequest(BaseModel):
    source: str = "data999"
    model: str
    prompt: str
    params: Optional[Dict[str, Any]] = {}


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
    body = {
        "模型id": model_id,
        "用户消息": user_msg,
        "渠道分组策略": "成功率优先",
        "对话组id": group_id,
        "生成参数": {"web_search": req.web_search, "enable_thinking": req.enable_thinking},
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


@app.get("/api/balance")
async def balance():
    r = await http().get(
        f"{DATA999_BASE}/v1/skills/balance",
        headers={"Authorization": f"Bearer {DATA999_KEY}"},
        timeout=10,
    )
    return r.json()


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
