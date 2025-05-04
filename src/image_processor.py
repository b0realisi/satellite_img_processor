# Importamos la librería OpenCV, que nos permite trabajar con imágenes fácilmente
import cv2

# Definimos la clase ImageProcessor
# Esta clase se encarga de cargar, mejorar y guardar imágenes
class ImageProcessor:
    
    # Constructor de la clase
    def __init__(self):
        pass  # Por ahora no necesita hacer nada especial al crearse el objeto

    # Método para cargar una imagen desde un archivo
    def load_image(self, image_path):
        """
        Carga una imagen en color desde la ruta especificada.
        """
        # Leemos la imagen usando OpenCV
        image = cv2.imread(image_path)

        # Verificamos que la imagen se haya cargado bien
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        # Devolvemos la imagen cargada
        return image

    # Método para guardar una imagen en disco
    def save_image(self, image, output_path):
        """
        Guarda la imagen procesada en la ruta especificada.
        """
        # Usamos OpenCV para guardar la imagen en la ruta de salida
        cv2.imwrite(output_path, image)

    # Método para aplicar ecualización de histograma a una imagen color
    def equalize_histogram_color(self, image):
        """
        Aplica ecualización de histograma a cada canal de color por separado (B, G, R).
        Esto mejora el contraste de la imagen.
        """

        # Dividimos la imagen en sus 3 canales de color: azul (B), verde (G) y rojo (R)
        channels = cv2.split(image)

        # Ecualizamos cada canal de forma independiente
        eq_channels = [cv2.equalizeHist(c) for c in channels]

        # Combinamos los canales ecualizados de nuevo en una imagen en color
        eq_image = cv2.merge(eq_channels)

        # Devolvemos la imagen resultante
        return eq_image

    # Método para aplicar un filtro de suavizado Gaussiano
    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        """
        Aplica un filtro de suavizado Gaussiano para reducir el ruido.
        kernel_size indica el tamaño del filtro (por defecto es 5x5).
        """

        # Aplicamos el filtro Gaussiano usando OpenCV
        blurred = cv2.GaussianBlur(image, kernel_size, 0)

        # Devolvemos la imagen suavizada
        return blurred
