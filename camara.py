import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "emotion_classifier.onnx"

CAMERA_INDEX = 0

WIDTH = 640
HEIGHT = 480

# Detectar rostros cada X frames
FACE_INTERVAL = 2

# Ejecutar el modelo de emociones cada X frames
EMOTION_INTERVAL = 3

# ============================================================
# EMOCIONES
# ============================================================

EMOTIONS = [
    "Enojado",
    "Disgusto",
    "Miedo",
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

# ============================================================
# COMPROBAR MODELO
# ============================================================

if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: No existe {MODEL_PATH}")
    raise SystemExit(1)

# ============================================================
# OPENCV
# ============================================================

cv2.setUseOptimized(True)

# Tu Ryzen tiene 4 núcleos / 8 hilos.
cv2.setNumThreads(4)

FACE_XML = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(FACE_XML)

if face_cascade.empty():
    print("ERROR: No se pudo cargar CascadeClassifier.")
    raise SystemExit(1)

print("CascadeClassifier: OK")

# ============================================================
# ONNX
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Modelo ONNX: OK")
print("Entrada:", session.get_inputs()[0].shape)
print("Salida:", session.get_outputs()[0].shape)

# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("CAP_DSHOW no funcionó.")

    cap.release()

    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: No se pudo abrir la cámara.")
    raise SystemExit(1)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

faces = []

current_emotion = "Detectando..."
current_confidence = 0.0

# Suavizado
emotion_history = []

HISTORY_SIZE = 5

# FPS
fps = 0.0

fps_frames = 0

fps_start = time.perf_counter()

# ============================================================
# INICIO
# ============================================================

print()
print("=" * 50)
print(" DETECTOR DE EMOCIONES OFFLINE")
print("=" * 50)
print("7 emociones")
print("Resolución:", WIDTH, "x", HEIGHT)
print("Presiona Q para salir.")
print("=" * 50)
print()

# ============================================================
# BUCLE
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: No se pudo leer la cámara.")
            break

        frame_count += 1
        fps_frames += 1

        # ----------------------------------------------------
        # ESCALA DE GRISES
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # DETECCIÓN FACIAL
        # ----------------------------------------------------

        if frame_count % FACE_INTERVAL == 0:

            small_gray = cv2.resize(
                gray,
                None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA
            )

            detected_faces = face_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24)
            )

            faces = [
                (
                    x * 2,
                    y * 2,
                    w * 2,
                    h * 2
                )
                for x, y, w, h
                in detected_faces
            ]

        # ----------------------------------------------------
        # PROCESAR ROSTROS
        # ----------------------------------------------------

        for x, y, w, h in faces:

            # Coordenadas seguras

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                frame.shape[1],
                x + w
            )

            y2 = min(
                frame.shape[0],
                y + h
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # ROSTRO RGB
            # ------------------------------------------------

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # MODELO
            # ------------------------------------------------

            if frame_count % EMOTION_INTERVAL == 0:

                # BGR -> RGB
                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                # 224x224
                face_rgb = cv2.resize(
                    face_rgb,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )

                # uint8 -> float32
                face_rgb = face_rgb.astype(
                    np.float32
                ) / 255.0

                # HWC -> CHW
                face_rgb = np.transpose(
                    face_rgb,
                    (2, 0, 1)
                )

                # Crear batch
                input_tensor = np.expand_dims(
                    face_rgb,
                    axis=0
                )

                # --------------------------------------------
                # INFERENCIA
                # --------------------------------------------

                output = session.run(
                    None,
                    {
                        input_name:
                        input_tensor
                    }
                )

                scores = output[0][0]

                # --------------------------------------------
                # SOFTMAX
                # --------------------------------------------

                scores = scores - np.max(scores)

                probabilities = (
                    np.exp(scores) /
                    np.sum(np.exp(scores))
                )

                emotion_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[emotion_index]
                )

                emotion = EMOTIONS[
                    emotion_index
                ]

                # --------------------------------------------
                # HISTORIAL
                # --------------------------------------------

                emotion_history.append(
                    emotion_index
                )

                if len(emotion_history) > HISTORY_SIZE:

                    emotion_history.pop(0)

                # La emoción más repetida
                counts = np.bincount(
                    emotion_history,
                    minlength=len(EMOTIONS)
                )

                stable_index = int(
                    np.argmax(counts)
                )

                current_emotion = EMOTIONS[
                    stable_index
                ]

                current_confidence = confidence

            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

            colors = {

                "Feliz":
                    (0, 255, 0),

                "Enojado":
                    (0, 0, 255),

                "Triste":
                    (255, 0, 0),

                "Sorpresa":
                    (0, 255, 255),

                "Miedo":
                    (255, 0, 255),

                "Disgusto":
                    (0, 128, 0),

                "Neutral":
                    (220, 220, 220)
            }

            color = colors.get(
                current_emotion,
                (255, 255, 255)
            )

            # ------------------------------------------------
            # RECTÁNGULO
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ------------------------------------------------
            # TEXTO
            # ------------------------------------------------

            text = (
                f"{current_emotion} "
                f"{current_confidence * 100:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.perf_counter()

        elapsed = (
            current_time -
            fps_start
        )

        if elapsed >= 1.0:

            fps = (
                fps_frames /
                elapsed
            )

            fps_frames = 0

            fps_start = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ====================================================
        # MOSTRAR
        # ====================================================

        cv2.imshow(
            "Detector de Emociones - Offline",
            frame
        )

        # Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    print("Cerrando...")

    cap.release()

    cv2.destroyAllWindows()