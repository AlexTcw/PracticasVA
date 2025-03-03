import cv2
import os

import numpy as np
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
    print("Converting image to gray")
    if image is None:
        raise ValueError("Error: La imagen no fue cargada correctamente.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

#convertir la imagen a una matriz de numeros
def image_to_matrix(image):
    print("Converting image to matrix")
    if image is None:
        print("Error: No hay imagen para mostrar.")
        return None
    else:
        return np.array(image)

def gaussian_filter(matrix, sigma):
    print("Gaussian filter")
    if matrix is None:
        print("Error: No hay imagen para mostrar.")
        return None

    # Definir el kernel gaussiano 5x5 con sigma = 1
    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]], dtype=np.float32)

    kernel /= kernel.sum()  # Normalizar el kernel

    height, width = matrix.shape
    filtered_matrix = np.zeros((height, width), dtype=np.uint8)  # Inicializar matriz de salida

    # Aplicar convolución manualmente
    for i in range(2, height - 2):  # Evitar bordes
        for j in range(2, width - 2):
            suma = 0
            for ki in range(-2, 3):
                for kj in range(-2, 3):
                    suma += matrix[i + ki, j + kj] * kernel[ki + 2, kj + 2]
            filtered_matrix[i, j] = int(suma)

    return filtered_matrix

def gaussian_filter_fast(matrix, sigma):
    print("Applying fast Gaussian filter")

    if matrix is None:
        print("Error: No hay imagen para mostrar.")
        return None

    # Aplicar el filtro Gaussiano con OpenCV
    filtered_matrix = cv2.GaussianBlur(matrix, (5, 5), sigma)

    return filtered_matrix

def compute_gradient(matrix):
    if matrix is None:
        print("Error: No hay imagen para procesar.")
        return None, None

    # Definir las máscaras de Sobel para los gradientes en X e Y
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    height, width = matrix.shape
    gradient_magnitude = np.zeros((height, width), dtype=np.float32)
    gradient_direction = np.zeros((height, width), dtype=np.float32)

    # Aplicar convolución manualmente
    for i in range(1, height - 1):  # Evitar bordes
        for j in range(1, width - 1):
            gx = 0
            gy = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    gx += matrix[i + ki, j + kj] * sobel_x[ki + 1, kj + 1]
                    gy += matrix[i + ki, j + kj] * sobel_y[ki + 1, kj + 1]

            # Calcular la magnitud y dirección del gradiente
            gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
            gradient_direction[i, j] = np.arctan2(gy, gx)  # Ángulo en radianes

    return gradient_magnitude, gradient_direction

def compute_gradient_fast(matrix):
    print("Applying fast gradient computation")

    if matrix is None:
        print("Error: No hay imagen para procesar.")
        return None, None

    # Calcular derivadas en X e Y con el operador Sobel
    grad_x = cv2.Sobel(matrix, cv2.CV_64F, 1, 0, ksize=3)  # Derivada en X
    grad_y = cv2.Sobel(matrix, cv2.CV_64F, 0, 1, ksize=3)  # Derivada en Y

    # Calcular magnitud del gradiente
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calcular dirección del gradiente (en radianes)
    gradient_direction = np.arctan2(grad_y, grad_x)

    return gradient_magnitude, gradient_direction

def apply_threshold(gradient_magnitude, low_threshold, high_threshold):
    print("Applying thresholding...")

    if gradient_magnitude is None:
        print("Error: No hay matriz de gradiente para procesar.")
        return None

    # Crear una matriz del mismo tamaño que el gradiente, inicializada en 0
    thresholded_matrix = np.zeros_like(gradient_magnitude)

    # Aplicar umbralización
    thresholded_matrix[gradient_magnitude >= high_threshold] = 255  # Bordes fuertes
    thresholded_matrix[
        (gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold)] = 127  # Bordes débiles
    thresholded_matrix[gradient_magnitude < low_threshold] = 0  # No bordes

    return thresholded_matrix

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    if gradient_magnitude is None or gradient_direction is None:
        print("Error: Gradiente no válido.")
        return None

    height, width = gradient_magnitude.shape
    suppressed_matrix = np.zeros((height, width), dtype=np.float32)

    # Convertir la dirección del gradiente de radianes a grados y ajustarla entre 0° y 180°
    gradient_direction = np.degrees(gradient_direction) % 180

    for i in range(1, height - 1):  # Evitar bordes
        for j in range(1, width - 1):
            q, r = 255, 255  # Valores predeterminados fuera de la imagen

            # Comparar con los píxeles vecinos en la dirección del gradiente
            if (0 <= gradient_direction[i, j] < 22.5) or (157.5 <= gradient_direction[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]  # Derecha
                r = gradient_magnitude[i, j - 1]  # Izquierda
            elif 22.5 <= gradient_direction[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]  # Abajo izquierda
                r = gradient_magnitude[i - 1, j + 1]  # Arriba derecha
            elif 67.5 <= gradient_direction[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]  # Abajo
                r = gradient_magnitude[i - 1, j]  # Arriba
            elif 112.5 <= gradient_direction[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]  # Arriba izquierda
                r = gradient_magnitude[i + 1, j + 1]  # Abajo derecha

            # Si el píxel actual es un máximo local, lo conservamos, si no, lo eliminamos (lo hacemos 0)
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed_matrix[i, j] = gradient_magnitude[i, j]
            else:
                suppressed_matrix[i, j] = 0

    return suppressed_matrix

def hysteresis_thresholding(suppressed_matrix, low_threshold, high_threshold):
    if suppressed_matrix is None:
        print("Error: La matriz suprimida no es válida.")
        return None

    height, width = suppressed_matrix.shape
    strong = 255  # Píxeles fuertes
    weak = 50  # Píxeles débiles (pueden volverse fuertes si están conectados)

    # Inicializar matriz con ceros
    thresholded_matrix = np.zeros((height, width), dtype=np.uint8)

    # Identificar píxeles fuertes y débiles
    strong_i, strong_j = np.where(suppressed_matrix >= high_threshold)
    weak_i, weak_j = np.where((suppressed_matrix >= low_threshold) & (suppressed_matrix < high_threshold))

    # Asignar valores
    thresholded_matrix[strong_i, strong_j] = strong
    thresholded_matrix[weak_i, weak_j] = weak

    # Conectar píxeles débiles a fuertes
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if thresholded_matrix[i, j] == weak:
                # Revisar si algún vecino es fuerte
                if (strong in thresholded_matrix[i - 1:i + 2, j - 1:j + 2]):
                    thresholded_matrix[i, j] = strong
                else:
                    thresholded_matrix[i, j] = 0  # Eliminar píxeles débiles no conectados

    return thresholded_matrix

def apply_prewitt(matrix):
    print("Applying Prewitt filter...")

    if matrix is None:
        print("Error: No hay matriz para procesar.")
        return None, None

    # Máscaras de Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    # Aplicar convolución manualmente
    grad_x = np.zeros_like(matrix, dtype=np.float64)
    grad_y = np.zeros_like(matrix, dtype=np.float64)

    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            grad_x[i, j] = np.sum(matrix[i - 1:i + 2, j - 1:j + 2] * prewitt_x)
            grad_y[i, j] = np.sum(matrix[i - 1:i + 2, j - 1:j + 2] * prewitt_y)

    return grad_x, grad_y

def apply_prewitt_fast(matrix):
    import cv2
    print("Applying Prewitt filter")

    if matrix is None:
        print("Error: No hay matriz para procesar.")
        return None, None

    # Aplicar filtros de Prewitt usando cv2
    grad_x = cv2.filter2D(matrix, cv2.CV_64F, np.array([[-1, 0, 1],
                                                        [-1, 0, 1],
                                                        [-1, 0, 1]]))

    grad_y = cv2.filter2D(matrix, cv2.CV_64F, np.array([[-1, -1, -1],
                                                        [0, 0, 0],
                                                        [1, 1, 1]]))

    return grad_x, grad_y

def apply_roberts(matrix):
    print("Applying Roberts filter (without cv2)...")

    if matrix is None:
        print("Error: No hay matriz para procesar.")
        return None, None

    # Máscaras de Roberts
    roberts_x = np.array([[1, 0],
                          [0, -1]])

    roberts_y = np.array([[0, 1],
                          [-1, 0]])

    # Aplicar convolución manualmente
    grad_x = np.zeros_like(matrix, dtype=np.float64)
    grad_y = np.zeros_like(matrix, dtype=np.float64)

    for i in range(matrix.shape[0] - 1):
        for j in range(matrix.shape[1] - 1):
            grad_x[i, j] = np.sum(matrix[i:i + 2, j:j + 2] * roberts_x)
            grad_y[i, j] = np.sum(matrix[i:i + 2, j:j + 2] * roberts_y)

    return grad_x, grad_y

def apply_roberts_fast(matrix):
    import cv2
    print("Applying Roberts filter (with cv2)...")

    if matrix is None:
        print("Error: No hay matriz para procesar.")
        return None, None

    # Aplicar filtros de Roberts usando cv2
    grad_x = cv2.filter2D(matrix, cv2.CV_64F, np.array([[1, 0],
                                                        [0, -1]]))

    grad_y = cv2.filter2D(matrix, cv2.CV_64F, np.array([[0, 1],
                                                        [-1, 0]]))

    return grad_x, grad_y

def matrix_to_image(matrix, output_path=None):
    if matrix is None:
        print("Error: La matriz no es válida.")
        return None

    # Normalizar la matriz si es de tipo float (convertir a valores entre 0 y 255)
    if matrix.dtype == np.float32 or matrix.dtype == np.float64:
        matrix = np.clip(matrix, 0, 255).astype(np.uint8)

    # Guardar la imagen si se especifica una ruta
    if output_path:
        cv2.imwrite(output_path, matrix)

    return matrix

def apply_robberts_filter(image, low_threshold=50, high_threshold=100):
    matrix = np.array(image, dtype=np.float64)
    print("Applying Prewitt filter")

    # Paso 1: Aplicar máscaras de Prewitt
    grad_x, grad_y = apply_roberts_fast(matrix)  # Devuelve dos matrices

    # Calcular magnitud del gradiente
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Umbralizar
    thresholded_matrix = apply_threshold(gradient_magnitude, low_threshold, high_threshold)

    return thresholded_matrix

def apply_prewitt_filter(image, low_threshold=50, high_threshold=100):
    matrix = np.array(image, dtype=np.float64)  # Convertir a matriz NumPy
    print("Applying Prewitt filter")

    # Paso 1: Aplicar máscaras de Prewitt
    grad_x, grad_y = apply_prewitt_fast(matrix)  # Devuelve dos matrices

    # Calcular magnitud del gradiente
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Umbralizar
    thresholded_matrix = apply_threshold(gradient_magnitude, low_threshold, high_threshold)

    return thresholded_matrix

def apply_sobel_filter(image, sigma=1, low_threshold=50, high_threshold=100):
    # Paso 1: Aplicar filtro Gaussiano
    blurred = gaussian_filter_fast(image,sigma)
    # Paso 2: Caluclar gradiente de intensidad
    print("Applying gradient...")
    gradient_magnitude, gradient_direction = compute_gradient(blurred)
    # Paso 3 Umbralizamos
    thresholded_matrix = apply_threshold(gradient_magnitude,low_threshold,high_threshold)

    return thresholded_matrix

def apply_canny_filter(image, sigma=1, low_threshold=50, high_threshold=100):

    # Paso 1: Aplicar filtro Gaussiano
        #blurred = gaussian_filter(image, sigma)
    blurred = gaussian_filter_fast(image, sigma)

    # Paso 2: Calcular gradiente de intensidad
    print("Applying gradient...")
        ##gradient_magnitude, gradient_direction = compute_gradient(blurred)
    gradient_magnitude, gradient_direction = compute_gradient_fast(blurred)

    # Paso 3: Supresión de no máximos
    print("Applying non maximum suppression...")
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Paso 4: Umbralización con histéresis
    print("Applying hysteresis threshold...")
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)

    return edges

if __name__ == '__main__':
    #1. Cargamos la imagen
    srcImg = load_image("tree_grayscale.png")
    #2. Convertimos a B/W
    srcBWImg = convert_to_gray(srcImg)
    #3. Convertimos a una matriz de datos
    srcMatrix = image_to_matrix(srcBWImg)
    ## Llamamos al filtro de interés
    if srcMatrix is not None:
        # dst = apply_canny_filter(srcMatrix)
        # dst = apply_sobel_filter(srcMatrix)
        # dst = apply_prewitt_filter(srcMatrix)
        dst = apply_robberts_filter(srcMatrix)
        dstImg = matrix_to_image(dst)
        show_images([(srcBWImg,"src",1000,600),(dstImg,"dst",1000,600)])

