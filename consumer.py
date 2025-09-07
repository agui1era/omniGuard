#!/usr/bin/env python3
# consumer.py — lee eventos, analiza y dispara alertas

import os
import time
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

# Config env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
PROMPT_ANALYSIS = os.getenv("PROMPT_ANALYSIS")

EVENT_LOG_FILE = os.getenv("EVENT_LOG_FILE", "events.log")
ALERT_SCORE_THRESHOLD = float(os.getenv("ALERT_SCORE_THRESHOLD", 0.5))
WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", 10))
ANALYZE_INTERVAL = int(os.getenv("ANALYZE_INTERVAL", 60))

# Telegram
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Sirena
ENABLE_SIREN = os.getenv("ENABLE_SIREN", "0") == "1"
SIREN_FILE = os.getenv("SIREN_FILE", "sirena.mp3")

# TTS
ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"
TTS_URL = os.getenv("TTS_URL")
TTS_MODEL = os.getenv("TTS_MODEL")
TTS_VOICE = os.getenv("TTS_VOICE")
TTS_LANG = os.getenv("TTS_LANG", "es")
TTS_MODE = os.getenv("TTS_MODE", "audio")

def read_events():
    if not os.path.exists(EVENT_LOG_FILE):
        return []
    with open(EVENT_LOG_FILE, "r") as f:
        lines = f.readlines()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)
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
        return {"score": 0.0, "text": f"Sin eventos recientes. Timestamp: {datetime.now(timezone.utc).isoformat()}"}

    prompt = f"{PROMPT_ANALYSIS}\nEventos:\n{json.dumps(events, ensure_ascii=False)}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un experto en monitoreo."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()
        raw = out["choices"][0]["message"]["content"].strip()
        data = json.loads(raw)
        return data
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

def play_siren():
    if ENABLE_SIREN and os.path.exists(SIREN_FILE):
        os.system(f"afplay {SIREN_FILE}")

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
        fname = "alerta.mp3" if TTS_MODE == "audio" else "alerta.ogg"
        with open(fname, "wb") as f:
            f.write(resp.content)

        if TTS_MODE == "audio":
            os.system(f"afplay {fname}")
        else:
            os.system(f"ffplay -nodisp -autoexit {fname}")
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
            play_siren()
            speak_text(msg)

        time.sleep(ANALYZE_INTERVAL)

if __name__ == "__main__":
    main()