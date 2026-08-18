import cv2
import os

FACE_XML = "haarcascade_frontalface_default.xml"

print("Archivo:", os.path.abspath(FACE_XML))
print("Existe:", os.path.isfile(FACE_XML))

cascade = cv2.CascadeClassifier(FACE_XML)

print("Cascade cargado:", not cascade.empty())