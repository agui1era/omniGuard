#!/usr/bin/env python3
# consumer.py — lee eventos, analiza y dispara alertas con Telegram + TTS

import os
import time
import json
import logging
import random
import subprocess
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
ANALYZE_INTERVAL = int(os.getenv("ANALYZE_INTERVAL", 7200))  # 2 horas por defecto para evitar rate limits

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

# Logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def validate_config():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PROMPT_ANALYSIS:
        missing.append("PROMPT_ANALYSIS")
    if ENABLE_TELEGRAM:
        if not TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise SystemExit(f"Faltan variables de entorno requeridas: {', '.join(missing)}")

def with_retries(request_fn, max_attempts=3, base_delay=1.0, max_delay=30.0):
    attempt = 0
    while True:
        try:
            return request_fn()
        except requests.RequestException as e:
            attempt += 1
            status = getattr(e.response, "status_code", None)
            retriable = isinstance(e, requests.Timeout) or (status in {429, 500, 502, 503, 504})
            if attempt >= max_attempts or not retriable:
                raise
            # Exponential backoff con jitter, más agresivo para 429
            if status == 429:
                sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)) * 2)  # Doble delay para rate limit
            else:
                sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.5 + random.random())
            logging.warning(f"Fallo intento {attempt}/{max_attempts} (status={status}). Reintentando en {sleep_s:.2f}s…")
            time.sleep(sleep_s)

def read_events(window_seconds: int | None = None):
    if not os.path.exists(EVENT_LOG_FILE):
        return []
    with open(EVENT_LOG_FILE, "r") as f:
        lines = f.readlines()
    effective_window = WINDOW_SECONDS if window_seconds is None else window_seconds
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=effective_window)
    events = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            ts_raw = str(data.get("timestamp"))
            # Manejar timestamps sin zona horaria (naive) y con zona horaria (aware)
            if ts_raw.endswith("Z"):
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            elif "+" in ts_raw or ts_raw.count("-") > 2:  # Tiene offset
                ts = datetime.fromisoformat(ts_raw)
            else:  # Timestamp naive, asumir UTC
                ts_naive = datetime.fromisoformat(ts_raw)
                ts = ts_naive.replace(tzinfo=timezone.utc)
            if ts > cutoff:
                events.append(data)
        except Exception as e:
            logging.warning(f"Error parseando evento: {e} -> {line.strip()[:200]}")
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
        "temperature": 0.2,
        # Solicitar JSON estructurado cuando el modelo lo soporte
        "response_format": {"type": "json_object"}
    }

    try:
        def _req():
            return requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
        r = with_retries(_req)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Intento de recuperación: encerrar en llaves si parece casi JSON
            content_fixed = content.strip()
            if not content_fixed.startswith("{"):
                content_fixed = "{" + content_fixed
            if not content_fixed.endswith("}"):
                content_fixed = content_fixed + "}"
            try:
                parsed = json.loads(content_fixed)
            except Exception:
                logging.error(f"Respuesta no-JSON del modelo: {content[:300]}")
                return {"score": 0.0, "text": "Respuesta del analizador no válida"}
        # Normalizar salida mínima
        score = float(parsed.get("score", 0.0))
        text = str(parsed.get("text", ""))
        return {"score": score, "text": text}
    except Exception as e:
        logging.error(f"Error analizando: {e}")
        return {"score": 0.0, "text": f"Error analizando: {e}"}

def send_telegram(msg):
    if not ENABLE_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        def _req():
            return requests.post(url, data=data, timeout=10)
        r = with_retries(_req)
        if not r.ok:
            logging.warning(f"Telegram respondió {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logging.error(f"Error enviando a Telegram: {e}")

def speak_text(text):
    if not ENABLE_TTS:
        return
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": TTS_MODEL, "voice": TTS_VOICE, "input": text}
        def _req():
            return requests.post(TTS_URL, headers=headers, json=payload, timeout=30)
        resp = with_retries(_req)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "audio" not in ctype:
            logging.warning(f"TTS Content-Type inesperado: {ctype} | body: {resp.text[:200]}")
        with open(TTS_OUTPUT, "wb") as f:
            f.write(resp.content)
        try:
            subprocess.run(["afplay", TTS_OUTPUT], check=False)
        except FileNotFoundError:
            # Fallback común en Linux si no existe afplay
            subprocess.run(["mpg123", TTS_OUTPUT], check=False)
    except Exception as e:
        logging.error(f"Error en TTS: {e}")

def main():
    validate_config()
    logging.info(f"Consumer online (umbral {ALERT_SCORE_THRESHOLD}–1)")

    # Llamada inicial con ventana fija de 1 hora (3600s)
    initial_events = read_events(window_seconds=3600)
    initial_result = analyze(initial_events)
    initial_score = initial_result.get("score", 0.0)
    initial_msg = initial_result.get("text", "")
    logging.info(f"[Inicial] Score={initial_score:.2f} | Msg={initial_msg}")
    if initial_score >= ALERT_SCORE_THRESHOLD:
        logging.warning("[Inicial] ALERTA!")
        send_telegram(f"🚨 ALERTA!\n{initial_msg}")
        speak_text(f"{TTS_MESSAGE}")
        time.sleep(10)
        speak_text(f"{TTS_MESSAGE}")

    while True:
        events = read_events()
        result = analyze(events)
        score = result.get("score", 0.0)
        msg = result.get("text", "")
        logging.info(f"Score={score:.2f} | Msg={msg}")

        if score >= ALERT_SCORE_THRESHOLD:
            logging.warning("ALERTA!")
            send_telegram(f"🚨 ALERTA!\n{msg}")
            speak_text(f"{TTS_MESSAGE}")
            time.sleep(10)
            speak_text(f"{TTS_MESSAGE}")

        time.sleep(ANALYZE_INTERVAL)

if __name__ == "__main__":
    main()