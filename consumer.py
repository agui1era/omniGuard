#!/usr/bin/env python3
# consumer.py — lee eventos, analiza y dispara alertas con Telegram + TTS

import os
import time
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

# Configuración base
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
PROMPT_ANALYSIS = os.getenv("PROMPT_ANALYSIS")

EVENT_LOG_FILE = os.getenv("EVENT_LOG_FILE", "events.log")
ALERT_SCORE_THRESHOLD = float(os.getenv("ALERT_SCORE_THRESHOLD", 0.5))
WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", 3600))  
ANALYZE_INTERVAL = int(os.getenv("ANALYZE_INTERVAL", 3600))

# Telegram
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# TTS
ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"
TTS_URL = os.getenv("TTS_URL", "https://api.openai.com/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "verse")
TTS_LANG = os.getenv("TTS_LANG", "es")
TTS_OUTPUT = os.getenv("TTS_OUTPUT", "alerta.mp3")
TTS_MESSAGE = os.getenv("TTS_MESSAGE", "Se ha detectado una alerta de seguridad")

def read_events():
    if not os.path.exists(EVENT_LOG_FILE):
        return []
    with open(EVENT_LOG_FILE, "r") as f:
        lines = f.readlines()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=WINDOW_SECONDS)
    events = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            if ts > cutoff:
                events.append(data)
        except Exception as e:
            print(f"⚠️ Error parseando evento: {e} -> {line.strip()}")
            continue
    return events

def analyze(events):
    if not events:
        return {"score": 0.0, "text": f"Sin eventos recientes. {datetime.now(timezone.utc).isoformat()}"}

    prompt = f"""{PROMPT_ANALYSIS}

Nota: YOLO puede fallar con falsos positivos o perder detecciones por perturbaciones de red.
Eventos:
{json.dumps(events, ensure_ascii=False)}
"""

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un sistema experto en monitoreo de seguridad. Sé breve, cauteloso y responde en JSON válido."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": 0.0, "text": f"Error analizando: {e}"}

def send_telegram(msg):
    if not ENABLE_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"⚠️ Error enviando a Telegram: {e}")

def speak_text(text):
    if not ENABLE_TTS:
        return
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        resp = requests.post(
            TTS_URL,
            headers=headers,
            json={"model": TTS_MODEL, "voice": TTS_VOICE, "input": text},
            timeout=30
        )
        resp.raise_for_status()
        with open(TTS_OUTPUT, "wb") as f:
            f.write(resp.content)
        os.system(f"afplay {TTS_OUTPUT}")  # en Linux usa mpg123 o ffplay
    except Exception as e:
        print(f"⚠️ Error en TTS: {e}")

def main():
    print(f"🚀 Consumer online (umbral {ALERT_SCORE_THRESHOLD}–1)")
    while True:
        events = read_events()
        result = analyze(events)
        score = result.get("score", 0.0)
        msg = result.get("text", "")
        print(f"📊 Score={score:.2f} | Msg={msg}")

        if score >= ALERT_SCORE_THRESHOLD:
            print("🚨 ALERTA!")
            send_telegram(f"🚨 ALERTA!\n{msg}")
            speak_text(f"{TTS_MESSAGE}")

        time.sleep(ANALYZE_INTERVAL)

if __name__ == "__main__":
    main()