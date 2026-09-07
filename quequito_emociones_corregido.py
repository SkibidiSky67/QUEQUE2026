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

# ============================================================
# CÁMARAS
# ============================================================

# El programa probará estas cámaras en orden:
# 0 -> 1 -> 2 -> 0 -> ...

CAMARAS = [0, 1, 2]

CAMARA_ACTUAL = 0

# ============================================================
# BLENDER
# ============================================================

BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = 5000

# PNG antiguo de Blender.
# Lo dejamos por compatibilidad, pero ahora las imágenes
# principales serán las 4 imágenes de emociones.
BLENDER_RENDER = r"C:\Users\Alumno_031\Documents\QUEQUE2026\queque_emocion.png"

# ============================================================
# 4 IMÁGENES DE EMOCIONES
# ============================================================

CARPETA_EMOCIONES = (
    r"C:\Users\Alumno_031\Documents\QUEQUE2026\emociones"
)

IMAGENES_EMOCIONES = {
    "Feliz": os.path.join(
        CARPETA_EMOCIONES,
        "FELIZ.png"
    ),

    "Triste": os.path.join(
        CARPETA_EMOCIONES,
        "TRISTE.png"
    ),

    "Sorpresa": os.path.join(
        CARPETA_EMOCIONES,
        "SORPRESA.png"
    ),

    "Neutral": os.path.join(
        CARPETA_EMOCIONES,
        "NEUTRAL.png"
    )
}

# ============================================================
# FRASES
# ============================================================

FRASES = {
    "Feliz":
        "Me alegra verte asi! Sigue disfrutando este momento.",

    "Triste":
        "Esta bien sentirse triste. Date un momento y recuerda que no estas solo.",

    "Sorpresa":
        "Vaya! Parece que algo te sorprendio.",

    "Neutral":
        "Todo tranquilo. Respira y sigue adelante."
}

# ============================================================
# EMOCIONES DEL MODELO
# ============================================================

EMOCIONES_MODELO = [
    "Enojado",
    "Disgusto",
    "Miedo",
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

# ============================================================
# EMOCIONES QUE MOSTRAMOS
# ============================================================

EMOCIONES = [
    "Feliz",
    "Triste",
    "Sorpresa",
    "Neutral"
]

INDICES = {
    "Feliz": 3,
    "Triste": 4,
    "Sorpresa": 5,
    "Neutral": 6
}

# ============================================================
# VARIABLES GLOBALES
# ============================================================

cap = None

ultima_emocion_blender = None

probabilidades_actuales = np.zeros(
    4,
    dtype=np.float32
)

probabilidades_suavizadas = (
    np.ones(4, dtype=np.float32) / 4
)

ALPHA = 0.15

emocion_actual = "Neutral"

mostrar_diagnostico = False

fps = 0
fps_contador = 0
fps_tiempo = time.time()

frame_contador = 0

PROCESAR_CADA = 2

ultimo_rostro = None

mostrar_resultado = False
solicitar_captura = False

captura_congelada = None
emocion_capturada = "Neutral"
probabilidad_capturada = 0.0

render_capturado = None

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
# COMPROBAR LAS 4 IMÁGENES
# ============================================================

print()
print("=" * 60)
print("COMPROBANDO IMÁGENES DE EMOCIONES")
print("=" * 60)

for emocion, ruta in IMAGENES_EMOCIONES.items():

    if os.path.exists(ruta):

        print(
            f"OK  - {emocion}: {ruta}"
        )

    else:

        print(
            f"FALTA - {emocion}: {ruta}"
        )

print("=" * 60)


# ============================================================
# CARGAR CALIBRACIÓN
# ============================================================

calibracion = np.load(
    CALIBRACION
)

promedio = calibracion["promedio"]
desviacion = calibracion["desviacion"]

print()
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

print()
print("Cargando modelo...")

sesion = ort.InferenceSession(
    MODELO,
    providers=[
        "CPUExecutionProvider"
    ]
)

entrada = sesion.get_inputs()[0].name

print("Modelo cargado correctamente.")
print(
    "Entrada:",
    entrada
)

print(
    "Forma:",
    sesion.get_inputs()[0].shape
)


# ============================================================
# HAAR CASCADE
# ============================================================

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)

if face_cascade.empty():

    print(
        "ERROR: No se pudo cargar Haar Cascade."
    )

    input("Presiona ENTER para salir...")
    exit()

print(
    "Haar Cascade cargado correctamente."
)


# ============================================================
# FUNCIÓN ABRIR CÁMARA
# ============================================================

def abrir_camara(indice):

    global cap

    print()
    print(
        "Intentando abrir cámara:",
        indice
    )

    # Cerrar cámara anterior
    if cap is not None:

        try:
            cap.release()
        except:
            pass

        cap = None

        time.sleep(0.5)

    # Windows suele funcionar mejor con CAP_DSHOW
    nueva_cap = cv2.VideoCapture(
        indice,
        cv2.CAP_DSHOW
    )

    if not nueva_cap.isOpened():

        print(
            "No se pudo abrir cámara:",
            indice
        )

        nueva_cap.release()

        return False

    nueva_cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    nueva_cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Comprobar que realmente entrega frames
    correcto = False

    for _ in range(10):

        ret, prueba = nueva_cap.read()

        if ret and prueba is not None:

            correcto = True
            break

        time.sleep(0.05)

    if not correcto:

        print(
            "La cámara abrió pero no entrega imagen."
        )

        nueva_cap.release()

        return False

    cap = nueva_cap

    print(
        "CÁMARA ABIERTA CORRECTAMENTE:",
        indice
    )

    return True


# ============================================================
# BUSCAR PRIMERA CÁMARA DISPONIBLE
# ============================================================

camara_encontrada = False

for indice in CAMARAS:

    if abrir_camara(indice):

        CAMARA_ACTUAL = indice

        camara_encontrada = True

        break


if not camara_encontrada:

    print()
    print(
        "ERROR: No se encontró ninguna cámara."
    )

    input("Presiona ENTER para salir...")
    exit()


# ============================================================
# CAMBIAR CÁMARA
# ============================================================

def cambiar_camara():

    global CAMARA_ACTUAL
    global ultimo_rostro
    global probabilidades_suavizadas
    global emocion_actual

    posicion = CAMARAS.index(
        CAMARA_ACTUAL
    )

    cantidad = len(CAMARAS)

    # Probar las siguientes cámaras
    for paso in range(
        1,
        cantidad + 1
    ):

        nueva_posicion = (
            posicion + paso
        ) % cantidad

        nuevo_indice = CAMARAS[
            nueva_posicion
        ]

        print()
        print("=" * 60)
        print(
            "CAMBIANDO CÁMARA:",
            CAMARA_ACTUAL,
            "->",
            nuevo_indice
        )
        print("=" * 60)

        if abrir_camara(
            nuevo_indice
        ):

            CAMARA_ACTUAL = nuevo_indice

            ultimo_rostro = None

            probabilidades_suavizadas = (
                np.ones(
                    4,
                    dtype=np.float32
                ) / 4
            )

            emocion_actual = "Neutral"

            print(
                "CAMBIO DE CÁMARA EXITOSO."
            )

            return True

    print(
        "No se pudo cambiar a otra cámara."
    )

    return False


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return (
        exp_x /
        np.sum(exp_x)
    )


# ============================================================
# CORREGIR SCORES
# ============================================================

def corregir_scores(scores):

    return (
        scores -
        promedio
    )


# ============================================================
# OBTENER PROBABILIDADES
# ============================================================

def obtener_probabilidades(scores):

    scores_corregidos = (
        corregir_scores(scores)
    )

    probabilidades_7 = softmax(
        scores_corregidos
    )

    seleccionadas = np.array(
        [
            probabilidades_7[
                INDICES["Feliz"]
            ],

            probabilidades_7[
                INDICES["Triste"]
            ],

            probabilidades_7[
                INDICES["Sorpresa"]
            ],

            probabilidades_7[
                INDICES["Neutral"]
            ]
        ],
        dtype=np.float32
    )

    suma = np.sum(
        seleccionadas
    )

    if suma > 0:

        seleccionadas /= suma

    return (
        scores_corregidos,
        probabilidades_7,
        seleccionadas
    )


# ============================================================
# ENVIAR A BLENDER
# ============================================================

def enviar_a_blender(emocion):

    emocion_blender = (
        emocion
        .lower()
        .strip()
    )

    print()
    print(
        "CONECTANDO CON BLENDER..."
    )

    try:

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as cliente:

            cliente.settimeout(
                8.0
            )

            cliente.connect(
                (
                    BLENDER_HOST,
                    BLENDER_PORT
                )
            )

            cliente.sendall(
                (
                    emocion_blender +
                    "\n"
                ).encode(
                    "utf-8"
                )
            )

            print(
                "MENSAJE ENVIADO A BLENDER:",
                emocion_blender
            )

            respuesta = (
                cliente.recv(
                    1024
                )
                .decode(
                    "utf-8"
                )
                .strip()
            )

        print(
            "RESPUESTA BLENDER:",
            repr(respuesta)
        )

        if respuesta.upper() == "OK":

            print(
                "BLENDER RESPONDIO OK."
            )

            return True

        return False

    except Exception as e:

        print(
            "ERROR BLENDER:",
            e
        )

        return False


# ============================================================
# CARGAR IMAGEN DE EMOCIÓN
# ============================================================

def cargar_imagen_emocion(
    emocion
):

    ruta = IMAGENES_EMOCIONES.get(
        emocion
    )

    if ruta is None:

        print(
            "No existe imagen para:",
            emocion
        )

        return None

    if not os.path.exists(ruta):

        print()
        print(
            "NO SE ENCUENTRA LA IMAGEN:"
        )

        print(ruta)

        return None

    imagen = cv2.imread(
        ruta,
        cv2.IMREAD_COLOR
    )

    if imagen is None:

        print(
            "ERROR AL CARGAR:",
            ruta
        )

        return None

    print()
    print(
        "IMAGEN CARGADA:",
        emocion
    )

    print(
        "RUTA:",
        ruta
    )

    print(
        "TAMAÑO:",
        imagen.shape
    )

    return imagen


# ============================================================
# TEXTO CENTRADO
# ============================================================

def texto_centrado(
    img,
    texto,
    y,
    escala=0.8,
    grosor=2
):

    h, w = img.shape[:2]

    (
        tw,
        th
    ), _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        grosor
    )

    x = max(
        10,
        (w - tw) // 2
    )

    cv2.putText(
        img,
        texto,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )


# ============================================================
# BOTÓN
# ============================================================

def dibujar_boton(
    img,
    x1,
    y1,
    x2,
    y2,
    texto
):

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (55, 55, 55),
        -1
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    (
        tw,
        th
    ), _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2
    )

    tx = (
        x1 +
        (x2 - x1 - tw) // 2
    )

    ty = (
        y1 +
        (y2 - y1 + th) // 2
    )

    cv2.putText(
        img,
        texto,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# CREAR PANTALLA RESULTADO
# ============================================================

def crear_pantalla_resultado(
    captura,
    emocion,
    probabilidad,
    imagen_emocion
):

    W = 1280
    H = 720

    pantalla = np.zeros(
        (H, W, 3),
        dtype=np.uint8
    )

    texto_centrado(
        pantalla,
        "RESULTADO DE TU EMOCION",
        55,
        1.05,
        2
    )

    # Panel izquierdo
    cv2.rectangle(
        pantalla,
        (30, 85),
        (620, 520),
        (35, 35, 35),
        -1
    )

    # Panel derecho
    cv2.rectangle(
        pantalla,
        (660, 85),
        (1250, 520),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        pantalla,
        "TU CAPTURA",
        (45, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        pantalla,
        "TU QUEQUITO",
        (675, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # CAPTURA DEL USUARIO
    # --------------------------------------------------------

    if captura is not None:

        imagen_camara = captura.copy()

        imagen_camara = cv2.resize(
            imagen_camara,
            (560, 370),
            interpolation=cv2.INTER_AREA
        )

        pantalla[
            135:505,
            60:620
        ] = imagen_camara

    # --------------------------------------------------------
    # IMAGEN DE LA EMOCIÓN
    # --------------------------------------------------------

    if imagen_emocion is not None:

        render = imagen_emocion.copy()

        rh, rw = render.shape[:2]

        escala = min(
            540 / rw,
            370 / rh
        )

        nw = max(
            1,
            int(rw * escala)
        )

        nh = max(
            1,
            int(rh * escala)
        )

        render = cv2.resize(
            render,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

        x = (
            955 -
            nw // 2
        )

        y = (
            320 -
            nh // 2
        )

        x = max(
            680,
            min(
                x,
                1230 - nw
            )
        )

        y = max(
            135,
            min(
                y,
                505 - nh
            )
        )

        pantalla[
            y:y + nh,
            x:x + nw
        ] = render

    else:

        texto_centrado(
            pantalla[
                135:505,
                660:1250
            ],
            "FALTA LA IMAGEN",
            190,
            0.65,
            2
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    texto_centrado(
        pantalla,
        f"EMOCION: {emocion.upper()}",
        565,
        0.95,
        2
    )

    texto_centrado(
        pantalla,
        f"PROBABILIDAD: {probabilidad * 100:.1f}%",
        605,
        0.65,
        2
    )

    frase = FRASES.get(
        emocion,
        ""
    )

    texto_centrado(
        pantalla,
        frase,
        645,
        0.58,
        1
    )

    dibujar_boton(
        pantalla,
        450,
        665,
        830,
        710,
        "NUEVA CAPTURA"
    )

    return pantalla


# ============================================================
# BARRAS
# ============================================================

def dibujar_barra(
    frame,
    nombre,
    probabilidad,
    x,
    y,
    ancho=220,
    alto=22
):

    texto = (
        f"{nombre}: "
        f"{probabilidad * 100:.1f}%"
    )

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

    cv2.rectangle(
        frame,
        (x, y),
        (x + ancho, y + alto),
        (60, 60, 60),
        -1
    )

    longitud = int(
        ancho * probabilidad
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + longitud, y + alto),
        (0, 200, 0),
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + ancho, y + alto),
        (255, 255, 255),
        1
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def dibujar_diagnostico(
    frame,
    scores,
    probabilidades
):

    alto, ancho = frame.shape[:2]

    x1 = 15
    y1 = 15
    x2 = min(
        ancho - 15,
        420
    )
    y2 = min(
        alto - 15,
        400
    )

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
            f"{i} - "
            f"{EMOCIONES_MODELO[i]:10s}: "
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
# VARIABLES DE DIAGNÓSTICO
# ============================================================

ultimo_scores = np.zeros(
    7,
    dtype=np.float32
)

ultimo_probabilidades_7 = np.zeros(
    7,
    dtype=np.float32
)


# ============================================================
# VENTANA
# ============================================================

NOMBRE_VENTANA = (
    "Quequito - Detector de Emociones"
)

cv2.namedWindow(
    NOMBRE_VENTANA,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    NOMBRE_VENTANA,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global mostrar_diagnostico
    global solicitar_captura
    global mostrar_resultado

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # --------------------------------------------------------
    # MODO RESULTADO
    # --------------------------------------------------------

    if mostrar_resultado:

        if (
            450 <= x <= 830
            and
            665 <= y <= 710
        ):

            mostrar_resultado = False
            solicitar_captura = False

        return

    # --------------------------------------------------------
    # CAPTURAR
    # --------------------------------------------------------

    if (
        450 <= x <= 830
        and
        650 <= y <= 710
    ):

        solicitar_captura = True
        return

    # --------------------------------------------------------
    # CAMBIAR CÁMARA
    # --------------------------------------------------------

    if (
        850 <= x <= 1030
        and
        650 <= y <= 710
    ):

        cambiar_camara()
        return

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------

    if (
        1050 <= x <= 1250
        and
        20 <= y <= 65
    ):

        mostrar_diagnostico = (
            not mostrar_diagnostico
        )


cv2.setMouseCallback(
    NOMBRE_VENTANA,
    mouse_callback
)


# ============================================================
# MENSAJE INICIAL
# ============================================================

print()
print("=" * 60)
print("QUEQUITO - DETECTOR DE EMOCIONES")
print("=" * 60)
print(
    "Cámara actual:",
    CAMARA_ACTUAL
)
print()
print(
    "Feliz / Triste / Sorpresa / Neutral"
)
print()
print(
    "C = capturar"
)
print(
    "D = diagnóstico"
)
print(
    "V = cambiar cámara"
)
print(
    "ESC = salir"
)
print("=" * 60)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    # ========================================================
    # MODO RESULTADO
    # ========================================================

    if mostrar_resultado:

        pantalla = crear_pantalla_resultado(
            captura_congelada,
            emocion_capturada,
            probabilidad_capturada,
            render_capturado
        )

        cv2.imshow(
            NOMBRE_VENTANA,
            pantalla
        )

        tecla = cv2.waitKey(30) & 0xFF

        if tecla == 27:
            break

        if tecla in (
            ord("n"),
            ord("N")
        ):

            mostrar_resultado = False

        continue

    # ========================================================
    # LEER CÁMARA
    # ========================================================

    if cap is None:

        if not abrir_camara(
            CAMARA_ACTUAL
        ):

            time.sleep(1)
            continue

    ret, frame = cap.read()

    if not ret or frame is None:

        print(
            "No se pudo leer cámara",
            CAMARA_ACTUAL
        )

        # Intentar otra cámara
        cambiar_camara()

        continue

    frame_contador += 1

    # Espejo
    frame = cv2.flip(
        frame,
        1
    )

    frame_para_captura = frame.copy()

    # ========================================================
    # DETECCIÓN FACIAL
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    rostros = (
        face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
    )

    if len(rostros) > 0:

        rostro = max(
            rostros,
            key=lambda r:
            r[2] * r[3]
        )

        x, y, w, h = rostro

        ultimo_rostro = rostro

        # ====================================================
        # PROCESAR ONNX
        # ====================================================

        if (
            frame_contador %
            PROCESAR_CADA == 0
        ):

            cara = frame[
                y:y+h,
                x:x+w
            ]

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

                cara = (
                    cara.astype(
                        np.float32
                    ) / 255.0
                )

                cara = np.transpose(
                    cara,
                    (2, 0, 1)
                )

                cara = np.expand_dims(
                    cara,
                    axis=0
                )

                # ============================================
                # INFERENCIA
                # ============================================

                salida = sesion.run(
                    None,
                    {
                        entrada: cara
                    }
                )[0][0]

                # ============================================
                # CALIBRACIÓN
                # ============================================

                (
                    scores_corregidos,
                    probabilidades_7,
                    probabilidades_4
                ) = obtener_probabilidades(
                    salida
                )

                ultimo_scores = (
                    scores_corregidos
                )

                ultimo_probabilidades_7 = (
                    probabilidades_7
                )

                probabilidades_actuales = (
                    probabilidades_4
                )

                # ============================================
                # SUAVIZADO
                # ============================================

                probabilidades_suavizadas = (
                    ALPHA *
                    probabilidades_actuales
                    +
                    (1 - ALPHA) *
                    probabilidades_suavizadas
                )

                # ============================================
                # EMOCIÓN FINAL
                # ============================================

                indice = np.argmax(
                    probabilidades_suavizadas
                )

                emocion_actual = (
                    EMOCIONES[indice]
                )

    else:

        ultimo_rostro = None

    # ========================================================
    # CAPTURA
    # ========================================================

    if solicitar_captura:

        captura_congelada = (
            frame_para_captura.copy()
        )

        emocion_capturada = (
            emocion_actual
        )

        probabilidad_capturada = float(
            probabilidades_suavizadas[
                EMOCIONES.index(
                    emocion_capturada
                )
            ]
        )

        print()
        print("=" * 60)
        print(
            "CAPTURA REALIZADA"
        )
        print(
            "EMOCION:",
            emocion_capturada.upper()
        )
        print(
            f"PROBABILIDAD: "
            f"{probabilidad_capturada * 100:.1f}%"
        )
        print("=" * 60)

        # ====================================================
        # ENVIAR A BLENDER
        # ====================================================

        enviar_a_blender(
            emocion_capturada
        )

        # ====================================================
        # CARGAR IMAGEN CORRESPONDIENTE
        # ====================================================

        render_capturado = (
            cargar_imagen_emocion(
                emocion_capturada
            )
        )

        if render_capturado is None:

            print()
            print(
                "ADVERTENCIA:"
            )

            print(
                "No se encontró la imagen de",
                emocion_capturada
            )

        else:

            print(
                "IMAGEN DE EMOCIÓN "
                "CARGADA CORRECTAMENTE."
            )

        mostrar_resultado = True

        solicitar_captura = False

        continue

    # ========================================================
    # REDIMENSIONAR
    # ========================================================

    frame_pantalla = cv2.resize(
        frame,
        (1280, 720),
        interpolation=cv2.INTER_LINEAR
    )

    # ========================================================
    # RECTÁNGULO ROSTRO
    # ========================================================

    if len(rostros) > 0:

        escala_x = (
            1280 /
            frame.shape[1]
        )

        escala_y = (
            720 /
            frame.shape[0]
        )

        dx = int(
            x * escala_x
        )

        dy = int(
            y * escala_y
        )

        dw = int(
            w * escala_x
        )

        dh = int(
            h * escala_y
        )

        cv2.rectangle(
            frame_pantalla,
            (dx, dy),
            (dx + dw, dy + dh),
            (0, 255, 0),
            3
        )

    # ========================================================
    # EMOCIÓN
    # ========================================================

    cv2.putText(
        frame_pantalla,
        emocion_actual,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    # ========================================================
    # BARRAS
    # ========================================================

    x_barra = 25
    y_barra = 85

    for i in range(4):

        dibujar_barra(
            frame_pantalla,
            EMOCIONES[i],
            probabilidades_suavizadas[i],
            x_barra,
            y_barra,
            ancho=300,
            alto=25
        )

        y_barra += 58

    # ========================================================
    # BOTÓN CAPTURAR
    # ========================================================

    dibujar_boton(
        frame_pantalla,
        450,
        650,
        640,
        705,
        "CAPTURAR"
    )

    # ========================================================
    # BOTÓN CAMBIAR CÁMARA
    # ========================================================

    dibujar_boton(
        frame_pantalla,
        660,
        650,
        840,
        705,
        "CAMBIAR CAM"
    )

    # ========================================================
    # BOTÓN DIAGNÓSTICO
    # ========================================================

    dibujar_boton(
        frame_pantalla,
        1050,
        20,
        1250,
        65,
        "DIAGNOSTICO"
    )

    # ========================================================
    # CÁMARA ACTUAL
    # ========================================================

    cv2.putText(
        frame_pantalla,
        f"CAMARA: {CAMARA_ACTUAL}",
        (1050, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # FPS
    # ========================================================

    fps_contador += 1

    tiempo_actual = time.time()

    if (
        tiempo_actual -
        fps_tiempo >= 1.0
    ):

        fps = (
            fps_contador /
            (
                tiempo_actual -
                fps_tiempo
            )
        )

        fps_contador = 0

        fps_tiempo = (
            tiempo_actual
        )

    cv2.putText(
        frame_pantalla,
        f"FPS: {fps:.1f}",
        (1100, 120),
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
            frame_pantalla,
            ultimo_scores,
            ultimo_probabilidades_7
        )

    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        NOMBRE_VENTANA,
        frame_pantalla
    )

    tecla = cv2.waitKey(1) & 0xFF

    # ========================================================
    # ESC
    # ========================================================

    if tecla == 27:
        break

    # ========================================================
    # C = CAPTURAR
    # ========================================================

    if tecla in (
        ord("c"),
        ord("C")
    ):

        solicitar_captura = True

    # ========================================================
    # V = CAMBIAR CÁMARA
    # ========================================================

    if tecla in (
        ord("v"),
        ord("V")
    ):

        cambiar_camara()

    # ========================================================
    # D = DIAGNÓSTICO
    # ========================================================

    if tecla in (
        ord("d"),
        ord("D")
    ):

        mostrar_diagnostico = (
            not mostrar_diagnostico
        )


# ============================================================
# CERRAR
# ============================================================

print()
print(
    "Cerrando cámara..."
)

if cap is not None:

    cap.release()

cv2.destroyAllWindows()

print(
    "Programa cerrado correctamente."
)