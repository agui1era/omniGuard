#!/usr/bin/env python3
# server.py — API de eventos + análisis (Python 3.9 compatible)

import os
import json
import re
import requests
import datetime as dt
from typing import Optional, List
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import threading
import time

load_dotenv()

# ===== Config (.env) =====
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4.1")

SYSTEM_PROMPT    = os.getenv(
    "SYSTEM_PROMPT",
    "Eres un sistema experto en seguridad. "
    "Debes responder EXCLUSIVAMENTE un JSON válido con claves: "
    "{\"score\": float entre 0 y 1, \"text\": string}. "
    "No incluyas nada fuera del objeto JSON."
)

PROMPT_ANALYSIS  = os.getenv(
    "PROMPT_ANALYSIS",
    "Analiza eventos y devuelve JSON {\"score\":float,\"text\":string}."
)

EVENT_LOG_FILE   = os.getenv("EVENT_LOG_FILE", "events.log")
LOG_CLEAN_DAYS   = int(os.getenv("LOG_CLEAN_DAYS", "30"))

# ===== App =====
app = FastAPI()

# ===== Modelos =====
class Event(BaseModel):
    source: str
    description: str
    value: Optional[float] = None
    timestamp: Optional[str] = None  # ISO8601 string

# ===== Utils =====
def now_iso() -> str:
    return dt.datetime.utcnow().isoformat()

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_event(ev: dict):
    ensure_dir_for(EVENT_LOG_FILE)
    with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def parse_iso(s: str) -> Optional[dt.datetime]:
    try:
        s2 = s.rstrip("Z")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None

def load_events(hours: int) -> List[dict]:
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=max(1, hours))
    items: List[dict] = []
    if not os.path.exists(EVENT_LOG_FILE):
        return items
    with open(EVENT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                ts_s = e.get("timestamp")
                if not ts_s:
                    continue
                ts = parse_iso(ts_s)
                if ts and ts >= cutoff:
                    items.append(e)
            except Exception:
                continue
    return items

def openai_analyze(events: List[dict]) -> dict:
    """
    Llama a /v1/chat/completions y devuelve {score, text}.
    Fuerza JSON y, si hay error, retorna el body para debug.
    """
    events_text = "\n".join(
        f"[{e.get('timestamp')}] {e.get('source')}: {e.get('description')} (valor={e.get('value')})"
        for e in events
    ) or "(sin eventos)"

    system_msg = SYSTEM_PROMPT
    user_msg   = f"{PROMPT_ANALYSIS}\n\nEventos:\n{events_text}"

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            body = r.text[:800]
            return {"score": 0.0, "text": f"OpenAI {r.status_code}: {body}"}

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return {"score": 0.0, "text": f"Respuesta inesperada: {str(data)[:400]}"}

        # Parseo robusto del JSON
        try:
            parsed = json.loads(content)
        except Exception as e:
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                return {"score": 0.0, "text": f"No vino JSON válido. Raw: {content[:200]}"}
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                return {"score": 0.0, "text": f"No pude parsear JSON: {e}. Raw: {content[:200]}"}

        score = float(parsed.get("score", 0.0))
        text  = parsed.get("text") or parsed.get("mensaje") or "Sin resumen"
        if score < 0: score = 0.0
        if score > 1: score = 1.0
        return {"score": score, "text": text}

    except Exception as e:
        return {"score": 0.0, "text": f"Error analizando: {e}"}

# ===== Limpieza de logs (cada 24h) =====
def cleanup_logs_once():
    try:
        if LOG_CLEAN_DAYS <= 0:
            return
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=LOG_CLEAN_DAYS)

        if os.path.isfile(EVENT_LOG_FILE):
            mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(EVENT_LOG_FILE))
            if mtime < cutoff:
                open(EVENT_LOG_FILE, "w").close()
                print(f"🧹 Truncado {EVENT_LOG_FILE} por antigüedad")

        logs_dir = "logs"
        if os.path.isdir(logs_dir):
            for name in os.listdir(logs_dir):
                p = os.path.join(logs_dir, name)
                if os.path.isfile(p):
                    mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(p))
                    if mtime < cutoff:
                        os.remove(p)
                        print(f"🧹 Borrado log viejo: {p}")

    except Exception as e:
        print(f"⚠ Error cleanup_logs_once: {e}")

def schedule_cleanup_daily():
    def _loop():
        while True:
            cleanup_logs_once()
            time.sleep(24*3600)
    threading.Thread(target=_loop, daemon=True).start()

# ===== Endpoints =====
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

@app.post("/event")
def add_event(ev: Event):
    data = ev.dict()
    if not data.get("timestamp"):
        data["timestamp"] = now_iso()
    save_event(data)
    return {"status": "stored"}

@app.get("/events")
def list_events():
    rows: List[dict] = []
    if os.path.exists(EVENT_LOG_FILE):
        with open(EVENT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return {"count": len(rows), "items": rows}

@app.get("/analyze")
def analyze(hours: int = Query(1, ge=1, le=168, description="Horas hacia atrás a analizar (1..168)")):
    events = load_events(hours)
    if not events:
        return {
            "status": "no_events",
            "score": 0.0,
            "msg": "Sin eventos recientes.",
            "events_count": 0,
            "window_hours": hours,
        }
    res = openai_analyze(events)
    return {
        "status": "ok",
        "score": float(res.get("score", 0.0)),
        "msg": res.get("text", "Sin resumen"),
        "events_count": len(events),
        "window_hours": hours,
    }

# ===== Main =====
if __name__ == "__main__":
    schedule_cleanup_daily()
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)