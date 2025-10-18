# monitor_riesgo.py
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
load_dotenv()  # Carga .env del directorio actual

LM_STUDIO_API = os.getenv("LM_STUDIO_API", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-vl-8b")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
INTERVAL = float(os.getenv("INTERVAL", 5))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
TIMEOUT = int(os.getenv("TIMEOUT", 60))
FRAME_MAX_WIDTH = int(os.getenv("FRAME_MAX_WIDTH", 960))  # 0 = desactivado

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", 0.0))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"

# =========================
# Utilidades
# =========================
def resize_if_needed(frame, max_width: int):
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def overlay_info(frame, texto: str, riesgo):
    """Dibuja info en pantalla: riesgo y primera línea del texto."""
    overlay = frame.copy()
    h = 28
    header = f"RISK={riesgo:.2f}" if isinstance(riesgo, (int, float)) else "RISK=?"
    primera_linea = (texto or "").splitlines()[0][:90] if texto else ""
    bar_h = 40
    # Color por nivel de riesgo
    if riesgo is None:
        color = (200, 200, 200)
    elif riesgo < 0.33:
        color = (80, 200, 120)   # verde
    elif riesgo < 0.66:
        color = (0, 200, 255)    # amarillo/azul
    else:
        color = (0, 0, 255)      # rojo

    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (30, 30, 30), -1)
    cv2.putText(overlay, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, primera_linea, (140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

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
    "Ejemplo válido:\n"
    "DESCRIPCION: Persona en pasillo con objeto metálico.\n"
    "RIESGOS: posible herramienta punzante; postura tensa\n"
    "ACCION: observar; si se aproxima, avisar a seguridad\n"
    "RISK=0.62"
)

def analizar_imagen(frame):
    im_b64, data_uri = a_b64_jpg(frame)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]},
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 700,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(LM_STUDIO_API, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content, im_b64
            else:
                print(f"[LLM] Intento {attempt}: HTTP {resp.status_code} -> {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[LLM] Intento {attempt}: Timeout tras {TIMEOUT}s")
        except Exception as e:
            print(f"[LLM] Intento {attempt}: Excepción: {e}")
        time.sleep(1)
    return "❌ Sin respuesta tras reintentos\nRISK=0.00", im_b64

# =========================
# Parsing de riesgo
# =========================
RISK_REGEX = re.compile(r"RISK\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)

def extraer_riesgo(texto: str):
    """
    Busca 'RISK=<decimal>' en cualquier parte del texto.
    Retorna float en [0,1] o None si no se encuentra.
    """
    if not texto:
        return None
    m = RISK_REGEX.search(texto)
    if not m:
        return None
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
        return None, "Credenciales Telegram no configuradas (.env)"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": desc[:1024]}
    files = {"photo": base64.b64decode(im_b64)}
    try:
        resp = requests.post(url, data=data, files=files, timeout=20)
        return resp.status_code, resp.text
    except Exception as e:
        return None, f"Error al enviar a Telegram: {e}"

# =========================
# Loop principal
# =========================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ No se puede abrir la cámara.")
        return

    print("✅ Cámara abierta. Presiona 'q' para salir.")
    ultimo_envio_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ No captura imagen.")
                break

            frame = resize_if_needed(frame, FRAME_MAX_WIDTH)

            # Analizar
            texto, im_b64 = analizar_imagen(frame)
            riesgo = extraer_riesgo(texto)
            print("──────── RESULTADO LLM ────────")
            print(texto)
            print("──────── RIESGO:", riesgo)

            # Overlay en ventana (opcional)
            vis = overlay_info(frame, texto, riesgo)
            if SHOW_WINDOW:
                cv2.imshow("Webcam - Risk Monitor", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Envío a Telegram si supera umbral y al menos INTERVAL entre envíos
            now = time.time()
            if riesgo is not None and riesgo >= RISK_THRESHOLD and (now - ultimo_envio_ts) >= INTERVAL:
                status, resp = enviar_telegram(im_b64, texto)
                print("📨 Telegram:", status, (resp[:160] + "...") if isinstance(resp, str) and len(resp) > 160 else resp)
                ultimo_envio_ts = now

            time.sleep(INTERVAL)

    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()