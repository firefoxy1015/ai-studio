import json
import re
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

CLAUDE_MODELS = {
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
}
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
    source: str = "data999"   # 'data999' | 'bytefor'
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


async def _stream_openai(req: ChatRequest):
    headers = {"Authorization": f"Bearer {DATA999_KEY}", "Content-Type": "application/json"}
    msgs: list = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend([{"role": m.role, "content": m.content} for m in req.messages])
    body = {"model": req.model, "stream": True, "messages": msgs}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{DATA999_BASE}/v1/chat/completions",
                                  headers=headers, json=body) as r:
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
                        yield f"data: {json.dumps({'text': text})}\n\n"
                except Exception:
                    pass
    yield "data: [DONE]\n\n"


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


# ── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.model in CLAUDE_MODELS:
        gen = _stream_claude(req)
    elif req.model in GEMINI_MODELS:
        gen = _stream_gemini(req)
    else:
        gen = _stream_openai(req)
    return StreamingResponse(
        gen, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat — collects full response and returns JSON."""
    if req.model in CLAUDE_MODELS:
        gen = _stream_claude(req)
    elif req.model in GEMINI_MODELS:
        gen = _stream_gemini(req)
    else:
        gen = _stream_openai(req)
    full = ""
    async for chunk in gen:
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
            try:
                import json as _json
                d = _json.loads(chunk[6:].strip())
                full += d.get("text", "")
            except Exception:
                pass
    return {"text": full}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    params = dict(req.params or {})

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
