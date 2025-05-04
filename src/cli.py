# Importamos argparse para manejar los argumentos de la línea de comandos
import argparse

# Importamos nuestras clases de procesamiento y extracción
from src.image_processor import ImageProcessor
from src.feature_extractor import FeatureExtractor

# Importamos OpenCV (solo para mostrar imágenes en caso de ser necesario)
import cv2

# Importamos matplotlib para mostrar resultados como imágenes y bordes
import matplotlib.pyplot as plt

# Definimos la función principal que controlará todo el flujo
def main():
    # Creamos un parser para leer los argumentos que pasamos desde la terminal
    parser = argparse.ArgumentParser(description="Procesamiento Digital de Imágenes Satelitales")

    # Definimos el primer argumento: qué queremos hacer
    parser.add_argument("command", choices=["process", "extract"], help="Comando a ejecutar") # El usuario debe elegir entre: process-> mejorar una imagen (ecualizar o suavizar) y extract-> extraer información de la imagen (histograma o bordes).
    # Ruta de la imagen de entrada (obligatorio)
    parser.add_argument("--input", type=str, required=True, help="Ruta de imagen de entrada") #--input: imagen que vas a procesar o analizar.
    # Ruta de salida de imagen procesada (opcional, solo en process)
    parser.add_argument("--output", type=str, help="Ruta de imagen de salida (solo para 'process')") #--output: dónde guardar la imagen si querés (en process).
    # Tipo de operación específica que queremos hacer
    parser.add_argument("--operation", choices=["equalize", "blur", "histogram", "edges"], required=True,
                        help="Operación a realizar") #--operation: qué tipo de procesamiento o extracción querés hacer.
    # Parseamos (leemos) los argumentos que el usuario pasó - convertimos lo que escribió el usuario en objetos que podemos usar en el código.
    args = parser.parse_args()

    # Instanciamos nuestros procesadores - creamos las instancias de las clases
    processor = ImageProcessor()
    extractor = FeatureExtractor()

    # Cargamos la imagen de entrada usando ImageProcessor
    image = processor.load_image(args.input) #cargamos la imagen desde la ruta que el usuario pasó como --input.

    #si el usuario pidió process (es decir, mejorar la imagen):
    if args.command == "process":
        if args.operation == "equalize": #aplicamos ecualización de histograma
            result = processor.equalize_histogram_color(image)
        elif args.operation == "blur": #aplicamos filtro Gaussiano (suavizado).
            result = processor.apply_gaussian_blur(image)
        else:
            raise ValueError("Operación inválida para 'process'") #si se ingreso mal la operación: --operation, tiramos un error

        # Si el usuario indicó una salida (--output), guardamos la imagen procesada
        if args.output:
            processor.save_image(result, args.output)
            print(f"Imagen procesada y guardada en {args.output}")
        # Si no puso --output, se muestra la imagen en pantalla
        else:
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title("Imagen Procesada")
            plt.axis("off")
            plt.show()

    # Si el usuario pidió extract (es decir, sacar información de la imagen):
    elif args.command == "extract":
        if args.operation == "histogram": # si se pidio histogram, se muestra el histograma de colores.
            extractor.calculate_histogram(image)
        elif args.operation == "edges": # si se pidió edges: detectamos bordes con Canny y mostramos el resultado en escala de grises.
            edges = extractor.detect_edges_canny(image)
            plt.imshow(edges, cmap='gray')
            plt.title("Bordes detectados")
            plt.axis("off")
            plt.show()
        else:
            raise ValueError("Operación inválida para 'extract'") # si se selecciono una opcion invalida, se mostrará un error

# Ejecutamos la función main() solo si se corre cli.py o main.py directamente
if __name__ == "__main__":
    main()
