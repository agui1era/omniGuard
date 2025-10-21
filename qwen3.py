# omni_guard.py
import os
import cv2
import time
import re
import base64
import requests
from dotenv import load_dotenv

# =========================
# Configuración y constantes
# =========================
load_dotenv()

LM_STUDIO_API = os.getenv("LM_STUDIO_API", "http://agentes.alabs.cl:9000/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-vl-8b")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
AUTO_CAMERA_SCAN = os.getenv("AUTO_CAMERA_SCAN", "1") == "1"
INTERVAL = float(os.getenv("INTERVAL", 5))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
TIMEOUT = int(os.getenv("TIMEOUT", 60))
FRAME_MAX_WIDTH = int(os.getenv("FRAME_MAX_WIDTH", 960))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", 0.8))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "0")

# =========================
# Utilidades
# =========================
def log(msg, color="\033[0m"):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}\033[0m")

def resize_if_needed(frame, max_width: int):
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def overlay_info(frame, texto: str, riesgo):
    overlay = frame.copy()
    header = f"RISK={riesgo:.2f}" if isinstance(riesgo, (int, float)) else "RISK=?"
    primera_linea = (texto or "").splitlines()[0][:90] if texto else ""
    bar_h = 40

    if riesgo is None:
        color = (200, 200, 200)
    elif riesgo < 0.33:
        color = (80, 200, 120)
    elif riesgo < 0.66:
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (30, 30, 30), -1)
    cv2.putText(overlay, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, primera_linea, (140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

def a_b64_jpg(frame):
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Fallo al codificar JPEG")
    im_b64 = base64.b64encode(buf).decode("utf-8")
    return im_b64, f"data:image/jpeg;base64,{im_b64}"

# =========================
# LLM: análisis de imagen
# =========================
SYSTEM_PROMPT = (
    "Eres un analista visual para detección de riesgo en seguridad física.\n"
    "Responde SIEMPRE en el siguiente formato, sin variarlo:\n\n"
    "DESCRIPCION: <una frase breve que describa la escena>\n"
    "RIESGOS: <lista breve de riesgos detectados u 'ninguno'>\n"
    "ACCION: <recomendación breve>\n"
    "RISK=<valor entre 0 y 1 con hasta 2 decimales>\n\n"
)

def analizar_imagen(frame):
    im_b64, data_uri = a_b64_jpg(frame)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]},
    ]
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 700}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(LM_STUDIO_API, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content, im_b64
            else:
                log(f"[LLM] HTTP {resp.status_code}: {resp.text[:150]}", "\033[33m")
        except requests.exceptions.Timeout:
            log(f"[LLM] Timeout tras {TIMEOUT}s", "\033[33m")
        except Exception as e:
            log(f"[LLM] Excepción: {e}", "\033[31m")
        time.sleep(1)

    return "❌ Sin respuesta tras reintentos\nRISK=0.00", im_b64

# =========================
# Parsing de riesgo
# =========================
RISK_REGEX = re.compile(r"RISK\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
def extraer_riesgo(texto: str):
    if not texto:
        return None
    m = RISK_REGEX.search(texto)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return None

# =========================
# Telegram
# =========================
def enviar_telegram(im_b64: str, desc: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None, "Credenciales Telegram no configuradas"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": desc[:1024]}
    files = {"photo": base64.b64decode(im_b64)}
    try:
        resp = requests.post(url, data=data, files=files, timeout=20)
        return resp.status_code, resp.text
    except Exception as e:
        return None, f"Error al enviar a Telegram: {e}"

# =========================
# Búsqueda de cámara y loop principal
# =========================
def buscar_camara():
    log("🎥 Buscando cámara disponible...", "\033[36m")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        time.sleep(1)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                log(f"✅ Cámara encontrada en índice {i}", "\033[32m")
                return cap
            cap.release()
    return None

def main():
    cap = None
    if AUTO_CAMERA_SCAN:
        cap = buscar_camara()
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap or not cap.isOpened():
        log("❌ No se encontró cámara funcional.", "\033[31m")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log("🎬 Iniciando captura. Ctrl+C para salir.", "\033[36m")

    ultimo_envio_ts = 0.0
    fail_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                log(f"⚠️ Fallo de captura ({fail_count})", "\033[33m")
                if fail_count >= 10:
                    log("🔄 Reiniciando búsqueda de cámara...", "\033[35m")
                    cap.release()
                    return main()
                time.sleep(1)
                continue

            fail_count = 0
            frame = resize_if_needed(frame, FRAME_MAX_WIDTH)

            texto, im_b64 = analizar_imagen(frame)
            riesgo = extraer_riesgo(texto)

            log("──────── RESULTADO LLM ────────", "\033[37m")
            print(texto)
            log(f"RIESGO DETECTADO: {riesgo}", "\033[32m" if riesgo and riesgo >= RISK_THRESHOLD else "\033[33m")

            now = time.time()
            if riesgo is not None and riesgo >= RISK_THRESHOLD and (now - ultimo_envio_ts) >= INTERVAL:
                status, resp = enviar_telegram(im_b64, texto)
                log(f"📨 Telegram: {status} {resp[:150]}", "\033[36m")
                ultimo_envio_ts = now

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        log("🛑 Captura finalizada por el usuario.", "\033[31m")
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()