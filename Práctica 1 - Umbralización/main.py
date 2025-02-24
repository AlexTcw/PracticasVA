import cv2
import os

import numpy as np

UMBRAL = 15
DEFAULT_IMAGE_PATH = "resources/img/"


# Cargar la imagen
def load_image(image_name):
    image_path = os.path.join(DEFAULT_IMAGE_PATH, image_name)
    print("Loading image:", image_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")

    return image


# Convertir a escala de grises
def convert_to_gray(image):
    if image is None:
        raise ValueError("Error: La imagen no fue cargada correctamente.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Mostrar imagen
def show_image(image, window_name="Imagen", width=1000, height=600):
    if image is None:
        print("Error: No hay imagen para mostrar.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Mostrar mas de una imagen
def show_images(images):
    if not images:
        print("Error: No hay imagenes para mostrar.")
        return
    for image, window_name, width, height in images:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image):
    #Creamos una imagen del mismo tamaño
    height, width = image.shape
    #matriz vacía de destino
    dst = np.zeros((height, width), dtype=np.uint8)
    #recorremos la imagen
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            #Umbralizamos si es mayor o igual al umbral es 255 si es menor es 0
            if pixel >= UMBRAL:
                dst[i, j] = 255
            else:
                dst[i, j] = 0
    return dst


if __name__ == '__main__':
    # 1. Cargamos la imagen
    srcImg = load_image("tree_grayscale.png")
    if srcImg is not None:
        #2. Nos aseguramos que es escala de grises
        grayImage = convert_to_gray(srcImg)
        #3. Umbralizamos
        dstImg = process_image(grayImage)
        #4. Mostramos el original y el resultado
        show_images([(srcImg,"src",1000,600), (dstImg,"dst",1000,600)])
