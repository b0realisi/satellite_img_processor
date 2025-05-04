# Importamos OpenCV para procesamiento de imágenes
import cv2

# Importamos NumPy para trabajar con matrices y datos numéricos
import numpy as np

# Importamos matplotlib.pyplot para graficar histograma y bordes
import matplotlib.pyplot as plt

# Definimos la clase FeatureExtractor
# Esta clase se encarga de extraer información importante de las imágenes
class FeatureExtractor:
    
    # Constructor de la clase
    def __init__(self):
        pass  # No necesitamos inicializar nada por ahora

    # Método para calcular y mostrar el histograma de color de una imagen
    def calculate_histogram(self, image):
        """
        Calcula y muestra el histograma de una imagen en color (canales B, G, R).
        """

        # Definimos los nombres de los canales de color: azul, verde y rojo
        color = ('b', 'g', 'r')

        # Iteramos sobre cada canal de color
        for i, col in enumerate(color):
            # Calculamos el histograma para el canal actual
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            
            # Graficamos el histograma con el color correspondiente
            plt.plot(hist, color=col)
            plt.xlim([0, 256])  # Limite del eje x: valores de intensidad de 0 a 255

        # Agregamos título y etiquetas al gráfico
        plt.title("Histograma de Color")
        plt.xlabel("Intensidad")
        plt.ylabel("Cantidad de píxeles")

        # Mostramos el gráfico
        plt.show()

    # Método para detectar bordes en la imagen usando el detector de Canny
    def detect_edges_canny(self, image, low_threshold=50, high_threshold=150):
        """
        Detecta bordes en una imagen usando el algoritmo de Canny.
        Retorna una imagen binaria (bordes blancos sobre fondo negro).
        """

        # Convertimos la imagen a escala de grises, porque Canny necesita una sola capa de intensidades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicamos el detector de bordes de Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Devolvemos la imagen de bordes detectados
        return edges
