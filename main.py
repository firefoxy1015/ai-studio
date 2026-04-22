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
# Models confirmed to work on data999 but not on deepwl — skip deepwl entirely
DATA999_ONLY_MODELS = {"grok-4.2", "grok-4.2-image", "doubao-seed-2-0-pro-260215"}

# data999 internal API (lingkeai) — better availability than public API for grok-4.2
LINGKEAI_BASE = "https://php.lingkeai.vip"
LINGKEAI_SESSION_TOKEN = "e5b7ae5474930aaba74e50025f263888"
LINGKEAI_USER_ID = "9011036"
LINGKEAI_S6 = "Chengchen@630"
LINGKEAI_MODEL_IDS = {"grok-4.2": 94, "claude-opus-4-7": 90}


def _encode_lingkeai_token() -> str:
    timestamp = int(time.time())
    n = (LINGKEAI_SESSION_TOKEN + "|" + str(timestamp)).encode("utf-8")
    key = LINGKEAI_S6.encode("utf-8")
    xored = bytes([n[i] ^ key[i % len(key)] for i in range(len(n))])
    return base64.b64encode(xored).decode("ascii")
GEMINI_MODELS = {
    "gemini-3.1-pro-preview", "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview",
}
# These models expect images as a single string, not an array
SINGLE_IMG_MODELS = {"sora-2-all", "grok-video-3", "grok-video-3-plus"}
# sora-2 uses a different param name
SORA2_MODEL = "sora-2"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[str] = None


class GenerateRequest(BaseModel):
    source: str = "data999"   # 'data999' | 'bytefor' | 'deepwl'
    model: str
    prompt: str
    params: Optional[Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("index.html").read_text(encoding="utf-8")


# ── Chat streaming helpers ───────────────────────────────────────────────────

async def _stream_claude(req: ChatRequest):
    headers = {"x-api-key": DATA999_KEY, "Content-Type": "application/json"}
    body = {
        "model": req.model,
        "max_tokens": 8192,
        "stream": True,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
    }
    if req.system:
        body["system"] = req.system
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{DATA999_BASE}/v1/messages",
                                  headers=headers, json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    d = json.loads(s)
                    if d.get("type") == "content_block_delta":
                        text = d.get("delta", {}).get("text", "")
                        if text:
                            yield f"data: {json.dumps({'text': text})}\n\n"
                except Exception:
                    pass
    yield "data: [DONE]\n\n"


async def _stream_openai_from(req: ChatRequest, base: str, key: str):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    msgs: list = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend([{"role": m.role, "content": m.content} for m in req.messages])
    body = {"model": req.model, "stream": True, "messages": msgs, "network": True, "search": True}
    got_text = False
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{base}/v1/chat/completions",
                                  headers=headers, json=body) as r:
            if r.status_code >= 400:
                raise Exception(f"HTTP {r.status_code}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    d = json.loads(s)
                    text = d["choices"][0]["delta"].get("content", "")
                    if text:
                        got_text = True
                        yield f"data: {json.dumps({'text': text})}\n\n"
                except Exception:
                    pass
    if not got_text:
        raise Exception("empty response")


async def _stream_openai(req: ChatRequest):
    """Try deepwl first, fall back to data999 on error."""
    buf: list[str] = []
    try:
        async for chunk in _stream_openai_from(req, DEEPWL_BASE, DEEPWL_KEY):
            buf.append(chunk)
            yield chunk
        return
    except Exception:
        pass
    # fallback: data999
    if buf:
        return  # already yielded some content, don't double-send
    async for chunk in _stream_openai_from(req, DATA999_BASE, DATA999_KEY):
        yield chunk


async def _stream_gemini(req: ChatRequest):
    headers = {"x-goog-api-key": DATA999_KEY, "Content-Type": "application/json"}
    msgs = []
    for m in req.messages:
        role = "model" if m.role == "assistant" else "user"
        msgs.append({"role": role, "parts": [{"text": m.content}]})
    body = {"contents": msgs}
    if req.system:
        body["system_instruction"] = {"parts": [{"text": req.system}]}
    url = f"{DATA999_BASE}/v1beta/models/{req.model}:streamGenerateContent"
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=body) as r:
            async for chunk in r.aiter_text():
                for match in re.finditer(r'"text":\s*"((?:[^"\\]|\\.)*)"', chunk):
                    raw = match.group(1)
                    text = (raw.replace("\\n", "\n").replace("\\t", "\t")
                            .replace('\\"', '"').replace("\\\\", "\\"))
                    if text.strip():
                        yield f"data: {json.dumps({'text': text})}\n\n"
    yield "data: [DONE]\n\n"


async def _chat_lingkeai(req: ChatRequest) -> str:
    """Call data999's internal API (php.lingkeai.vip) for grok-4.2.
    Uses web_search=true which routes through a less-loaded backend."""
    model_id = LINGKEAI_MODEL_IDS.get(req.model)
    if not model_id:
        raise Exception(f"No lingkeai model ID for {req.model}")

    # Format full conversation as a single string for stateless operation
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
        "生成参数": {"web_search": True},
    }

    full = ""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{LINGKEAI_BASE}/moxing/tongyirukouchat",
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
                    # Claude format: {"content":"...","type":"content"}
                    if d.get("type") == "content" and d.get("content"):
                        full += d["content"]
                    # grok/OpenAI format: {"choices":[{"delta":{"content":"..."}}]}
                    choices = d.get("choices", [])
                    if choices:
                        text = choices[0].get("delta", {}).get("content", "")
                        if text:
                            full += text
                except json.JSONDecodeError:
                    pass
    if not full:
        raise Exception("lingkeai returned empty response")
    return full


# ── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.model in CLAUDE_MODELS:
        gen = _stream_claude(req)
    else:
        # deepwl supports OpenAI-compatible format for all models (incl. Gemini)
        # _stream_openai already handles deepwl-first + data999 fallback
        gen = _stream_openai(req)
    return StreamingResponse(
        gen, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat — try deepwl first, fall back to data999."""
    msgs: list = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend([{"role": m.role, "content": m.content} for m in req.messages])

    # claude-opus-4-7: use lingkeai (better availability)
    if req.model == "claude-opus-4-7":
        try:
            text = await _chat_lingkeai(req)
            if text:
                return {"text": text}
        except Exception:
            pass
        # fallback to data999 Anthropic format

    # Other Claude models use data999 Anthropic format
    if req.model in CLAUDE_MODELS:
        gen = _stream_claude(req)
        full = ""
        async for chunk in gen:
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    d = json.loads(chunk[6:].strip())
                    full += d.get("text", "")
                except Exception:
                    pass
        return {"text": full}

    # deepwl uses different model IDs in some cases
    DEEPWL_ID_MAP = {
        "grok-4.2": "grok-4-2",
        "grok-4.2-image": "grok-4-2-image",
    }

    # Non-Claude, non-data999-only: try deepwl first (30s timeout to fail fast)
    if req.model not in DATA999_ONLY_MODELS:
        deepwl_model = DEEPWL_ID_MAP.get(req.model, req.model)
        try:
            headers = {"Authorization": f"Bearer {DEEPWL_KEY}", "Content-Type": "application/json"}
            body = {"model": deepwl_model, "stream": False, "messages": msgs, "network": True, "search": True}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{DEEPWL_BASE}/v1/chat/completions", headers=headers, json=body)
            if r.status_code == 200:
                d = r.json()
                if not d.get("error"):
                    text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if text:
                        return {"text": text}
        except Exception:
            pass

    # DATA999_ONLY_MODELS: try lingkeai internal API first (better routing), then data999 public
    if req.model in DATA999_ONLY_MODELS:
        if req.model in LINGKEAI_MODEL_IDS:
            try:
                text = await _chat_lingkeai(req)
                if text:
                    return {"text": text}
            except Exception:
                pass

        # fallback: data999 public API non-streaming
        last_err = "no attempts"
        for attempt in range(3):
            try:
                if attempt > 0:
                    await asyncio.sleep(2)
                headers = {"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"}
                body = {"model": req.model, "stream": False, "messages": msgs}
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(f"{DATA999_BASE}/v1/chat/completions", headers=headers, json=body)
                if r.status_code == 200:
                    d = r.json()
                    if not d.get("error"):
                        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if text:
                            return {"text": text}
                        last_err = f"attempt {attempt}: empty content, raw={repr(r.text[:200])}"
                    else:
                        last_err = f"attempt {attempt}: error={d['error']}"
                else:
                    last_err = f"attempt {attempt}: HTTP {r.status_code}"
            except Exception as e:
                last_err = f"attempt {attempt}: {type(e).__name__}: {e}"
        raise HTTPException(503, detail=f"模型 {req.model} 失败: {last_err}")

    # data999 fallback for other models, retry up to 3 times (streaming)
    ERROR_SIGNALS = ("ResourceExhausted", "load_shed", "Stream aborted",
                     "upstream_error", "channel not found", "connect to sampling engine")
    for attempt in range(3):
        try:
            if attempt > 0:
                await asyncio.sleep(2)
            full = ""
            async for chunk in _stream_openai_from(req, DATA999_BASE, DATA999_KEY):
                if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    try:
                        d = json.loads(chunk[6:].strip())
                        full += d.get("text", "")
                    except Exception:
                        pass
            if full and not any(s in full for s in ERROR_SIGNALS):
                return {"text": full}
        except Exception:
            pass

    raise HTTPException(503, detail=f"模型 {req.model} 暂时不可用，请稍后重试或切换其他模型")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    params = dict(req.params or {})

    if req.source == "deepwl":
        headers = {"Authorization": f"Bearer {DEEPWL_KEY}", "Content-Type": "application/json"}
        body = {
            "model": req.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{DEEPWL_BASE}/v1/chat/completions", headers=headers, json=body)
        d = r.json()
        if d.get("error"):
            raise HTTPException(400, detail=d["error"].get("message", str(d["error"])))
        content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Prefer video URLs, fall back to any media URL
        video_exts = r'\.(?:mp4|webm|mov|m3u8)'
        urls = re.findall(rf'https?://[^\s\)\"\']+{video_exts}[^\s\)\"\']*', content)
        if not urls:
            # try markdown links for any extension
            all_links = re.findall(r'(?<!!)\[.*?\]\((https?://[^\)]+)\)', content)
            urls = all_links
        if not urls:
            raise HTTPException(400, detail=content or "deepwl 未返回媒体链接")
        return {"result_urls": urls, "source": "deepwl"}

    if req.source == "bytefor":
        headers = {"Authorization": f"Bearer {BYTEFOR_KEY}", "Content-Type": "application/json"}
        body = {"prompt": req.prompt, "model": req.model, **params}
        async with httpx.AsyncClient(timeout=35) as client:
            r = await client.post(f"{BYTEFOR_BASE}/api/v1/generate", headers=headers, json=body)
        d = r.json()
        if d.get("code") != 0:
            raise HTTPException(400, detail=d.get("msg", str(d)))
        return {"task_id": d["data"]["taskCode"], "source": "bytefor"}

    # DATA999 — fix images param format for certain models
    if "images" in params:
        imgs = params["images"]
        if isinstance(imgs, list):
            if req.model in SINGLE_IMG_MODELS:
                params["images"] = imgs[0] if imgs else ""
            elif req.model == SORA2_MODEL:
                params["input_reference"] = imgs[0] if imgs else ""
                del params["images"]

    headers = {"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"}
    body = {"model": req.model, "prompt": req.prompt, "params": params, "count": 1}
    async with httpx.AsyncClient(timeout=35) as client:
        r = await client.post(f"{DATA999_BASE}/v1/media/generate", headers=headers, json=body)
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
        headers = {"Authorization": f"Bearer {BYTEFOR_KEY}"}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{BYTEFOR_BASE}/api/v1/task/{task_id}", headers=headers)
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

    headers = {"Authorization": f"Bearer {DATA999_KEY}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{DATA999_BASE}/v1/skills/task-status?task_id={task_id}", headers=headers
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
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (file.filename, content,
                            file.content_type or "application/octet-stream")},
        )
    url = r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    return {"url": url, "name": file.filename}


@app.get("/api/balance")
async def balance():
    headers = {"Authorization": f"Bearer {DATA999_KEY}"}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{DATA999_BASE}/v1/skills/balance", headers=headers)
    return r.json()


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
