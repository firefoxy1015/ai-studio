import asyncio
import httpx
import json

DATA999_KEY = "sk-37b060cd778ee075ac3388fe421c6df1cc367f591238195c"
DATA999_BASE = "https://api.ai6700.com"
DEEPWL_KEY = "sk-hUviZm3xQzam0EaaA9622c041aA249CbB4924c929c9805Aa"
DEEPWL_BASE = "https://zx1.deepwl.net"

async def fetch_models(name, base_url, key):
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{base_url}/v1/models", headers=headers)
        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            print(f"--- {name} ({len(models)} models) ---")
            
            # Filter interesting ones
            interesting = []
            for m in models:
                m_lower = m.lower()
                if "search" in m_lower or "net" in m_lower or "grok" in m_lower or "doubao" in m_lower or "claude" in m_lower or "gemini" in m_lower or "gpt" in m_lower or "sonar" in m_lower or "perplexity" in m_lower:
                    interesting.append(m)
            
            print("Interesting models:", ", ".join(interesting[:50]))
            print("\nSearch/Net related models:")
            search_models = [m for m in models if "search" in m.lower() or "net" in m.lower() or "sonar" in m.lower() or "online" in m.lower() or "web" in m.lower()]
            print(", ".join(search_models))
            
            with open(f"{name}_models.json", "w", encoding="utf-8") as f:
                json.dump(models, f, ensure_ascii=False, indent=2)
        else:
            print(f"{name} Error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"{name} Exception: {e}")

async def main():
    await asyncio.gather(
        fetch_models("data999", DATA999_BASE, DATA999_KEY),
        fetch_models("deepwl", DEEPWL_BASE, DEEPWL_KEY)
    )

asyncio.run(main())
