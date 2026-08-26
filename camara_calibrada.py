import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELO = "emotion_classifier.onnx"
CALIBRACION = "calibracion.npz"

CAMARA = 0

EMOCIONES = [
    "Enojado",
    "Disgusto",
    "Miedo",
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

if not os.path.exists(MODELO):
    print("ERROR: No se encuentra emotion_classifier.onnx")
    input("ENTER para salir...")
    exit()

if not os.path.exists(CALIBRACION):
    print("ERROR: No se encuentra calibracion.npz")
    input("ENTER para salir...")
    exit()

# ============================================================
# CARGAR CALIBRACIÓN
# ============================================================

calibracion = np.load(CALIBRACION)

promedio = calibracion["promedio"]
desviacion = calibracion["desviacion"]

print("=" * 60)
print("CALIBRACIÓN")
print("=" * 60)

for i in range(7):
    print(
        f"{i} - {EMOCIONES[i]:10s} "
        f"promedio={promedio[i]:+.6f} "
        f"desv={desviacion[i]:.6f}"
    )

print("=" * 60)

# ============================================================
# CARGAR MODELO ONNX
# ============================================================

sesion = ort.InferenceSession(
    MODELO,
    providers=["CPUExecutionProvider"]
)

entrada = sesion.get_inputs()[0].name

print("Modelo:", MODELO)
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
    input("ENTER para salir...")
    exit()

# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(CAMARA)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    input("ENTER para salir...")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================================
# VARIABLES
# ============================================================

# Probabilidades iniciales
probabilidades = np.ones(7, dtype=np.float32) / 7.0

# Suavizado temporal
probabilidades_suavizadas = probabilidades.copy()

# Mientras más pequeño, más suave
ALPHA = 0.12

# Emoción mostrada
emocion_actual = "Neutral"

# Últimos scores
scores_originales = np.zeros(7, dtype=np.float32)
scores_corregidos = np.zeros(7, dtype=np.float32)

# Diagnóstico
mostrar_diagnostico = False

# FPS
fps = 0
fps_contador = 0
fps_tiempo = time.time()

# Procesar ONNX cada 2 frames para conservar FPS
frame_contador = 0
PROCESAR_CADA = 2

# ============================================================
# FUNCIONES
# ============================================================

def softmax(scores):

    scores = scores - np.max(scores)

    exp_scores = np.exp(scores)

    return exp_scores / np.sum(exp_scores)


def corregir_scores(scores):

    # Eliminamos el sesgo medido durante la calibración
    corregidos = scores - promedio

    return corregidos


def dibujar_barra(
    frame,
    nombre,
    probabilidad,
    x,
    y,
    ancho=250,
    alto=24
):

    # Texto
    texto = f"{nombre}: {probabilidad * 100:.1f}%"

    cv2.putText(
        frame,
        texto,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Fondo
    cv2.rectangle(
        frame,
        (x, y),
        (x + ancho, y + alto),
        (55, 55, 55),
        -1
    )

    # Barra
    longitud = int(ancho * float(probabilidad))

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


def dibujar_diagnostico(frame):

    alto, ancho = frame.shape[:2]

    x1 = 300
    y1 = 60
    x2 = min(ancho - 10, 630)
    y2 = min(alto - 10, 455)

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
        (x1 + 15, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Scores
    y = y1 + 55

    cv2.putText(
        frame,
        "Scores corregidos:",
        (x1 + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    y += 23

    for i in range(7):

        texto = (
            f"{i} - {EMOCIONES[i]:10s}: "
            f"{scores_corregidos[i]:+.4f}"
        )

        cv2.putText(
            frame,
            texto,
            (x1 + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        y += 21

    y += 5

    cv2.putText(
        frame,
        "Probabilidades:",
        (x1 + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    y += 22

    for i in range(7):

        texto = (
            f"{EMOCIONES[i]:10s}: "
            f"{probabilidades_suavizadas[i] * 100:5.2f}%"
        )

        cv2.putText(
            frame,
            texto,
            (x1 + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        y += 20


def mouse_callback(event, x, y, flags, param):

    global mostrar_diagnostico

    if event == cv2.EVENT_LBUTTONDOWN:

        # Botón diagnóstico
        if 470 <= x <= 625 and 10 <= y <= 50:

            mostrar_diagnostico = not mostrar_diagnostico


# ============================================================
# VENTANA
# ============================================================

cv2.namedWindow("Detector de emociones")

cv2.setMouseCallback(
    "Detector de emociones",
    mouse_callback
)

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

print()
print("=" * 60)
print("DETECTOR DE 7 EMOCIONES CALIBRADO")
print("=" * 60)
print()
print("Emociones:")
print("  0 - Enojado")
print("  1 - Disgusto")
print("  2 - Miedo")
print("  3 - Feliz")
print("  4 - Triste")
print("  5 - Sorpresa")
print("  6 - Neutral")
print()
print("ESC = salir")
print("Botón DIAGNOSTICO = ver información")
print("=" * 60)

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR leyendo cámara.")
        break

    # Espejo
    frame = cv2.flip(frame, 1)

    frame_contador += 1

    # ========================================================
    # DETECCIÓN DE ROSTRO
    # ========================================================

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
        x, y, w, h = max(
            rostros,
            key=lambda r: r[2] * r[3]
        )

        # ====================================================
        # INFERENCIA
        # ====================================================

        if frame_contador % PROCESAR_CADA == 0:

            cara = frame[y:y+h, x:x+w]

            if cara.size > 0:

                cara = cv2.resize(
                    cara,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )

                cara = cv2.cvtColor(
                    cara,
                    cv2.COLOR_BGR2RGB
                )

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

                # =================================================
                # MODELO
                # =================================================

                salida = sesion.run(
                    None,
                    {entrada: cara}
                )[0][0]

                scores_originales = salida.copy()

                # =================================================
                # CALIBRACIÓN
                # =================================================

                scores_corregidos = corregir_scores(
                    scores_originales
                )

                # =================================================
                # PROBABILIDADES
                # =================================================

                probabilidades = softmax(
                    scores_corregidos
                )

                # =================================================
                # SUAVIZADO
                # =================================================

                probabilidades_suavizadas = (
                    ALPHA * probabilidades
                    +
                    (1.0 - ALPHA) *
                    probabilidades_suavizadas
                )

                # Normalizar
                suma = np.sum(
                    probabilidades_suavizadas
                )

                if suma > 0:

                    probabilidades_suavizadas /= suma

                # =================================================
                # EMOCIÓN
                # =================================================

                indice = int(
                    np.argmax(
                        probabilidades_suavizadas
                    )
                )

                emocion_actual = EMOCIONES[indice]

        # ========================================================
        # RECTÁNGULO
        # ========================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    # ========================================================
    # EMOCIÓN PRINCIPAL
    # ========================================================

    cv2.putText(
        frame,
        emocion_actual,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    # ========================================================
    # BARRAS DE LAS 7 EMOCIONES
    # ========================================================

    x_barra = 20
    y_barra = 70

    for i in range(7):

        dibujar_barra(
            frame,
            EMOCIONES[i],
            probabilidades_suavizadas[i],
            x_barra,
            y_barra,
            ancho=250,
            alto=21
        )

        y_barra += 45

    # ========================================================
    # BOTÓN DIAGNÓSTICO
    # ========================================================

    cv2.rectangle(
        frame,
        (470, 10),
        (625, 50),
        (70, 70, 70),
        -1
    )

    cv2.rectangle(
        frame,
        (470, 10),
        (625, 50),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "DIAGNOSTICO",
        (482, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # FPS
    # ========================================================

    fps_contador += 1

    ahora = time.time()

    if ahora - fps_tiempo >= 1.0:

        fps = fps_contador / (
            ahora - fps_tiempo
        )

        fps_contador = 0
        fps_tiempo = ahora

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (500, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    if mostrar_diagnostico:

        dibujar_diagnostico(frame)

    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "Detector de emociones",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

# ============================================================
# CERRAR
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Programa cerrado correctamente.")