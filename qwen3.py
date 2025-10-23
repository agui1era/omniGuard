
import os
import cv2
import time
import re
import base64
import requests
import sys
from dotenv import load_dotenv

# =========================
# Configuración
# =========================
load_dotenv()

LM_STUDIO_API = os.getenv("LM_STUDIO_API", "http://agentes.alabs.cl:8888/v1/chat/completions")
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

# =========================
# Utilidades
# =========================
def log(msg, color="\033[0m"):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}\033[0m", flush=True)

def resize_if_needed(frame, max_width: int):
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

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
    "Prompt: Monitoreo y Cuidado de Adulto Mayor"
    "Eres un asistente de vigilancia y cuidado de adultos mayores."  
    "Analizas imágenes en tiempo real para detectar presencia, actividad, caídas u otras situaciones inusuales, y envías alertas automáticas a los cuidadores con la imagen y descripción del evento."

    "Instrucciones:"  
    "- Describe brevemente lo que ocurre en la imagen."  
    "- Indica si el adulto mayor está de pie, sentado, acostado o ausente."  
    "- Señala si parece necesitar ayuda (caída, desorientación, inactividad prolongada)."  
    "- Si detectas riesgo físico o de salud, marca la alerta como prioritaria."  
    "- Usa un tono empático y profesional."

    "Formato de salida:"  
    "Descripción: <qué ocurre en la escena>"  
    "Evaluación: <actividad normal o posible riesgo>"  
    "Alerta: <enviar a cuidadores si corresponde>"  
    "RISK=<valor entre 0.0 y 1.0>"
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
            log(f"[LLM] Timeout tras {TIMEOUT}s (intento {attempt})", "\033[33m")
        except requests.exceptions.RequestException as e:
            log(f"[LLM] Error de conexión: {e}", "\033[31m")
        time.sleep(2)

    return None, im_b64

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
            return val if 0.0 <= val <= 1.0 else None
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
# Cámara + bucle principal
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

def reiniciar_programa():
    log("🔄 Reiniciando programa...", "\033[35m")
    os.execv(sys.executable, ['python'] + sys.argv)

def main():
    cap = buscar_camara() if AUTO_CAMERA_SCAN else cv2.VideoCapture(CAMERA_INDEX)
    if not cap or not cap.isOpened():
        log("❌ No se encontró cámara funcional.", "\033[31m")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log("🎬 Iniciando captura (Ctrl+C para salir)", "\033[36m")

    fail_count = 0
    internet_failures = 0
    start_time = time.time()
    ultimo_envio_ts = 0.0

    try:
        while True:
            # Reinicio automático cada 24 horas
            if time.time() - start_time >= 86400:
                log("⏰ 24h cumplidas, reiniciando.", "\033[35m")
                reiniciar_programa()

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
            if texto is None:
                internet_failures += 1
                log(f"🌐 Error de conexión {internet_failures}/20", "\033[33m")
                if internet_failures >= 20:
                    log("💥 Pérdida de conexión persistente, reiniciando sistema.", "\033[31m")
                    reiniciar_programa()
                continue

            riesgo = extraer_riesgo(texto)
            internet_failures = 0

            log("──────── RESULTADO LLM ────────", "\033[37m")
            print(texto)
            log(f"RIESGO DETECTADO: {riesgo}", "\033[32m" if riesgo and riesgo >= RISK_THRESHOLD else "\033[33m")

            now = time.time()
            if riesgo is not None and riesgo >= RISK_THRESHOLD and (now - ultimo_envio_ts) >= INTERVAL:
                status, resp = enviar_telegram(im_b64, texto)
                log(f"📨 Telegram: {status} {resp[:120]}", "\033[36m")
                ultimo_envio_ts = now

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        log("🛑 Captura finalizada por el usuario.", "\033[31m")
    finally:
        cap.release()

if __name__ == "__main__":
    main()