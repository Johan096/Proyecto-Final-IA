import logging
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image

# Configuración básica de logging para ver mensajes informativos en la terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model(device: str):
    """
    Inicializa y retorna el processor y modelo BLIP para generación de captions.
    
    Args:
        device (str): Dispositivo a utilizar ('cuda' o 'cpu').
    
    Returns:
        processor: El processor BLIP.
        model: El modelo BLIP ya movido al dispositivo indicado.
    """
    logger.info("Cargando el processor y el modelo BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    logger.info("Modelo cargado correctamente.")
    return processor, model

def generate_caption(image: Image.Image, processor, model, device: str) -> str:
    """
    Genera un caption para una imagen utilizando el modelo BLIP.
    
    Args:
        image (PIL.Image.Image): Imagen de entrada.
        processor: Processor BLIP para preprocesar la imagen.
        model: Modelo BLIP para la generación de captions.
        device (str): Dispositivo en el que se ejecuta la inferencia.
    
    Returns:
        str: Caption generado para la imagen.
    """
    # Asegurarse de que la imagen sea un objeto PIL.Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Procesar la imagen indicando explícitamente el argumento "images"
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generar el caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    """
    Función principal que configura y lanza la interfaz de Gradio para el generador de captions.
    """
    # Configurar el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilizando dispositivo: {device}")

    # Inicializar el processor y modelo
    processor, model = initialize_model(device)

    # Configurar la interfaz de Gradio
    interface = gr.Interface(
        fn=lambda image: generate_caption(image, processor, model, device),
        inputs=gr.Image(type="pil", label="Sube una imagen"),
        outputs=gr.Textbox(label="Caption generado"),
        title="Generador de Captions con BLIP",
        description=(
            "Esta aplicación utiliza el modelo BLIP de Salesforce para generar descripciones "
            "para imágenes subidas. Simplemente carga una imagen y el modelo generará un caption descriptivo."
        ),
        article=(
            "Desarrollado como demostración del uso de modelos de deep learning en tareas de visión por computadora. "
            "Para más información, visita la documentación oficial de BLIP."
        ),
        allow_flagging="never"  # Desactiva la opción de marcar imágenes si no es necesaria
    )

    # Lanzar la interfaz en un servidor local
    interface.launch()

if __name__ == "__main__":
    main()
