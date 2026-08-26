import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import socket

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELO = "emotion_classifier.onnx"
CALIBRACION = "calibracion.npz"

CAMARA = 0

# ============================================================
# BLENDER
# ============================================================

BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = 5000

FRASES = {
    "Feliz": "¡Me alegra verte asi! Sigue disfrutando este momento.",
    "Triste": "Esta bien sentirse triste. Date un momento y recuerda que no estas solo.",
    "Sorpresa": "¡Vaya! Parece que algo te sorprendio.",
    "Neutral": "Todo tranquilo. Respira y sigue adelante."
}

ultima_emocion_blender = None

# Las 7 salidas REALES del modelo
EMOCIONES_MODELO = [
    "Enojado",
    "Disgusto",
    "Miedo",
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

# Emociones que mostraremos al usuario
EMOCIONES = [
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

# Índices correspondientes a las emociones anteriores
INDICES = {
    "Feliz": 3,
    "Triste": 4,
    "Sorpresa": 5,
    "Neutral": 6
}

# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

if not os.path.exists(MODELO):
    print("ERROR: No se encuentra:")
    print(MODELO)
    input("Presiona ENTER para salir...")
    exit()

if not os.path.exists(CALIBRACION):
    print("ERROR: No se encuentra:")
    print(CALIBRACION)
    input("Presiona ENTER para salir...")
    exit()

# ============================================================
# CARGAR CALIBRACIÓN
# ============================================================

calibracion = np.load(CALIBRACION)

promedio = calibracion["promedio"]
desviacion = calibracion["desviacion"]

print("=" * 60)
print("CALIBRACIÓN CARGADA")
print("=" * 60)

for i in range(7):
    print(
        f"{i} - {EMOCIONES_MODELO[i]:10s}: "
        f"promedio={promedio[i]:+.6f} "
        f"desv={desviacion[i]:.6f}"
    )

print("=" * 60)

# ============================================================
# CARGAR ONNX
# ============================================================

print("\nCargando modelo...")

sesion = ort.InferenceSession(
    MODELO,
    providers=["CPUExecutionProvider"]
)

entrada = sesion.get_inputs()[0].name

print("Modelo cargado correctamente.")
print("Entrada:", entrada)
print("Forma:", sesion.get_inputs()[0].shape)

# ============================================================
# HAAR CASCADE
# ============================================================

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: No se pudo cargar Haar Cascade.")
    input("Presiona ENTER para salir...")
    exit()

print("Haar Cascade cargado correctamente.")

# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(CAMARA)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    input("Presiona ENTER para salir...")
    exit()

# Intentar mantener una resolución razonable
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================================
# VARIABLES
# ============================================================

probabilidades_actuales = np.zeros(4, dtype=np.float32)

# Suavizado temporal
probabilidades_suavizadas = np.ones(4, dtype=np.float32) / 4

ALPHA = 0.15

emocion_actual = "Neutral"

# Diagnóstico
mostrar_diagnostico = False

# FPS
fps = 0
fps_contador = 0
fps_tiempo = time.time()

# Procesar modelo cada ciertos frames
frame_contador = 0

PROCESAR_CADA = 2

# Último rostro detectado
ultimo_rostro = None

# ============================================================
# FUNCIONES
# ============================================================

def softmax(x):
    """
    Convierte scores del modelo en probabilidades.
    """
    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


def corregir_scores(scores):
    """
    Corrige el sesgo obtenido durante la calibración.

    Restamos el promedio de cada salida para que una emoción
    que tenga un sesgo permanente no domine el resultado.
    """

    scores_corregidos = scores - promedio

    return scores_corregidos


def obtener_probabilidades(scores):
    """
    Aplica calibración + softmax.
    """

    scores_corregidos = corregir_scores(scores)

    probabilidades_7 = softmax(scores_corregidos)

    # Nos quedamos con:
    # Feliz, Triste, Sorpresa y Neutral

    seleccionadas = np.array([
        probabilidades_7[INDICES["Feliz"]],
        probabilidades_7[INDICES["Triste"]],
        probabilidades_7[INDICES["Sorpresa"]],
        probabilidades_7[INDICES["Neutral"]]
    ], dtype=np.float32)

    # Renormalizar las 4 emociones mostradas
    suma = np.sum(seleccionadas)

    if suma > 0:
        seleccionadas /= suma

    return (
        scores_corregidos,
        probabilidades_7,
        seleccionadas
    )


def enviar_a_blender(emocion):
    global ultima_emocion_blender

    emocion_blender = emocion.lower().strip()

    if emocion_blender == ultima_emocion_blender:
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cliente:
            cliente.settimeout(1.0)
            cliente.connect((BLENDER_HOST, BLENDER_PORT))
            cliente.sendall(emocion_blender.encode("utf-8"))

        ultima_emocion_blender = emocion_blender
        print("BLENDER <-", emocion.upper())

    except Exception as e:
        print("BLENDER NO DISPONIBLE:", e)


def dibujar_barra(
    frame,
    nombre,
    probabilidad,
    x,
    y,
    ancho=220,
    alto=22
):

    # Texto
    texto = f"{nombre}: {probabilidad * 100:.1f}%"

    cv2.putText(
        frame,
        texto,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Fondo
    cv2.rectangle(
        frame,
        (x, y),
        (x + ancho, y + alto),
        (60, 60, 60),
        -1
    )

    # Barra
    longitud = int(ancho * probabilidad)

    cv2.rectangle(
        frame,
        (x, y),
        (x + longitud, y + alto),
        (0, 200, 0),
        -1
    )

    # Borde
    cv2.rectangle(
        frame,
        (x, y),
        (x + ancho, y + alto),
        (255, 255, 255),
        1
    )


def dibujar_diagnostico(frame, scores, probabilidades):

    alto, ancho = frame.shape[:2]

    # Panel
    x1 = 15
    y1 = 15
    x2 = min(ancho - 15, 420)
    y2 = min(alto - 15, 400)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0,
        frame
    )

    cv2.putText(
        frame,
        "DIAGNOSTICO",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Scores corregidos:",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    y = 100

    for i in range(7):

        texto = (
            f"{i} - {EMOCIONES_MODELO[i]:10s}: "
            f"{scores[i]:+.4f}"
        )

        cv2.putText(
            frame,
            texto,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        y += 23

    y += 8

    cv2.putText(
        frame,
        "Probabilidades:",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    y += 25

    for i in range(4):

        nombre = EMOCIONES[i]

        texto = (
            f"{nombre:10s}: "
            f"{probabilidades[i] * 100:.2f}%"
        )

        cv2.putText(
            frame,
            texto,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        y += 22


# ============================================================
# BOTÓN DIAGNÓSTICO
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global mostrar_diagnostico

    if event == cv2.EVENT_LBUTTONDOWN:

        # Botón ubicado arriba a la derecha
        if 470 <= x <= 625 and 15 <= y <= 55:

            mostrar_diagnostico = not mostrar_diagnostico


# ============================================================
# CREAR VENTANA
# ============================================================

cv2.namedWindow("Detector de emociones")

cv2.setMouseCallback(
    "Detector de emociones",
    mouse_callback
)

# ============================================================
# BUCLE PRINCIPAL
# ============================================================

print()
print("=" * 60)
print("DETECTOR DE EMOCIONES")
print("=" * 60)
print("Feliz / Triste / Sorpresa / Neutral")
print()
print("Haz clic en DIAGNOSTICO para ver los scores.")
print("Presiona ESC para salir.")
print("=" * 60)

ultimo_scores = np.zeros(7, dtype=np.float32)
ultimo_probabilidades_7 = np.zeros(7, dtype=np.float32)

while True:

    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer la cámara.")
        break

    frame_contador += 1

    # Voltear como espejo
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # DETECCIÓN FACIAL
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    rostros = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(rostros) > 0:

        # Rostro más grande
        rostro = max(
            rostros,
            key=lambda r: r[2] * r[3]
        )

        x, y, w, h = rostro

        ultimo_rostro = rostro

        # ----------------------------------------------------
        # PROCESAR ONNX
        # ----------------------------------------------------

        if frame_contador % PROCESAR_CADA == 0:

            cara = frame[y:y+h, x:x+w]

            if cara.size > 0:

                # Redimensionar
                cara = cv2.resize(
                    cara,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )

                # BGR -> RGB
                cara = cv2.cvtColor(
                    cara,
                    cv2.COLOR_BGR2RGB
                )

                # Normalización
                cara = cara.astype(
                    np.float32
                ) / 255.0

                # HWC -> CHW
                cara = np.transpose(
                    cara,
                    (2, 0, 1)
                )

                # Batch
                cara = np.expand_dims(
                    cara,
                    axis=0
                )

                # ------------------------------------------------
                # INFERENCIA
                # ------------------------------------------------

                salida = sesion.run(
                    None,
                    {entrada: cara}
                )[0][0]

                ultimo_scores = salida.copy()

                # ------------------------------------------------
                # CALIBRACIÓN
                # ------------------------------------------------

                (
                    scores_corregidos,
                    probabilidades_7,
                    probabilidades_4
                ) = obtener_probabilidades(salida)

                ultimo_scores = scores_corregidos
                ultimo_probabilidades_7 = probabilidades_7

                probabilidades_actuales = probabilidades_4

                # ------------------------------------------------
                # SUAVIZADO
                # ------------------------------------------------

                probabilidades_suavizadas = (
                    ALPHA * probabilidades_actuales
                    +
                    (1 - ALPHA) *
                    probabilidades_suavizadas
                )

                # ------------------------------------------------
                # EMOCIÓN FINAL
                # ------------------------------------------------

                indice = np.argmax(
                    probabilidades_suavizadas
                )

                nueva_emocion = EMOCIONES[indice]

                if nueva_emocion != emocion_actual:
                    emocion_actual = nueva_emocion
                    enviar_a_blender(emocion_actual)

        # ----------------------------------------------------
        # RECTÁNGULO DEL ROSTRO
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    else:

        ultimo_rostro = None

    # ========================================================
    # TEXTO EMOCIÓN PRINCIPAL
    # ========================================================

    cv2.putText(
        frame,
        emocion_actual,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    # ========================================================
    # FRASE DE ANIMO
    # ========================================================

    frase_actual = FRASES.get(emocion_actual, "")

    cv2.putText(
        frame,
        frase_actual,
        (20, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # BARRAS
    # ========================================================

    x_barra = 20
    y_barra = 75

    for i in range(4):

        dibujar_barra(
            frame,
            EMOCIONES[i],
            probabilidades_suavizadas[i],
            x_barra,
            y_barra
        )

        y_barra += 48

    # ========================================================
    # BOTÓN DIAGNÓSTICO
    # ========================================================

    cv2.rectangle(
        frame,
        (470, 15),
        (625, 55),
        (80, 80, 80),
        -1
    )

    cv2.rectangle(
        frame,
        (470, 15),
        (625, 55),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "DIAGNOSTICO",
        (480, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # FPS
    # ========================================================

    fps_contador += 1

    tiempo_actual = time.time()

    if tiempo_actual - fps_tiempo >= 1.0:

        fps = fps_contador / (
            tiempo_actual - fps_tiempo
        )

        fps_contador = 0
        fps_tiempo = tiempo_actual

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (500, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    if mostrar_diagnostico:

        dibujar_diagnostico(
            frame,
            ultimo_scores,
            ultimo_probabilidades_7
        )

    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "Detector de emociones",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    # ESC
    if tecla == 27:
        break

# ============================================================
# CERRAR
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Programa cerrado correctamente.")