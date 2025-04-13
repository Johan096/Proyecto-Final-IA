## Nombre: Johan Daniel Muñoz salas 
## Matricula: 22-sisn-2-041
## Proyecto: Clasificador de imagenes 


import logging
import torch
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    BlipProcessor, 
    BlipForConditionalGeneration
)
import gradio as gr
from PIL import Image
import openai
import os
import time
from dotenv import load_dotenv

# Habilitar benchmark de cuDNN para acelerar la inferencia en GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Configura la API key de OpenAI usando la variable de entorno
openai.api_key =  os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función de comunicación con OpenAI para traducciones
def LLM(prompt_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # O usa "gpt-4" si tienes acceso
            messages=[
                {"role": "system", "content": "Eres un traductor profesional. Traduce el siguiente texto al español de forma precisa y natural."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        time.sleep(5)
        logger.error(f"Error de conexión al traducir: {e}")
        return LLM(prompt_text)

def translate_to_spanish(text):
    prompt = f"Traduce el siguiente texto al español:\n\n{text}"
    translation = LLM(prompt)
    return translation

def initialize_classifier(device: str):
    """
    Inicializa el modelo de clasificación ViT.
    """
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224").to(device)
    model.eval()
    return processor, model

def initialize_captioner(device: str):
    """
    Inicializa el modelo de captioning BLIP.
    """
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    caption_model.eval()
    return caption_processor, caption_model

def generate_caption(image, caption_processor, caption_model, device):
    """
    Genera una descripción en inglés utilizando BLIP.
    Se utiliza el prompt "a photo" (ó cualquier otro) para obtener un caption de calidad.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    # Utilizamos "a photo" para que el modelo genere una descripción en inglés
    inputs = caption_processor(image, text="a photo", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = caption_model.generate(**inputs, max_length=50)
    caption_en = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption_en

def classify_image(image, clf_processor, clf_model, caption_processor, caption_model, device):
    """
    Realiza la clasificación con ViT y obtiene un caption con BLIP.
    Luego, traduce el caption al español.
    """
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        
        # Clasificación con ViT
        inputs = clf_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clf_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        label = clf_model.config.id2label[top_idx[0].item()]
        confidence = top_prob[0].item() * 100

        # Diccionario de traducciones para etiquetas
        traducciones = {
            "cat": "gato",
            "dog": "perro",
            "person": "persona",
            "car": "auto",
            "bicycle": "bicicleta",
            "truck": "camión",
            "bird": "pájaro",
            "boat": "barco"
        }
        label_es = traducciones.get(label.lower(), label)
        resultado = f"{label_es.capitalize()}: {confidence:.2f}%"
        
        # Generación de descripción en inglés y traducción al español
        caption_en = generate_caption(image, caption_processor, caption_model, device)
        caption_es = translate_to_spanish(caption_en)
        
        return image, resultado, caption_es
    except Exception as e:
        logger.error(f"Error al clasificar o generar descripción: {e}")
        return None, "Error al clasificar la imagen.", "No se pudo generar la descripción."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inicializar modelos
    clf_processor, clf_model = initialize_classifier(device)
    caption_processor, caption_model = initialize_captioner(device)
    
    # Estilos personalizados para la interfaz
    css_estilizado = """
        .custom-header {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .custom-description {
            font-size: 1rem;
            margin-bottom: 10px;
            text-align: center;
            color: #555;
        }
        .custom-button {
            padding: 10px 20px;
            font-size: 1rem;
            border-radius: 5px;
        }
    """
    
    with gr.Blocks(css=css_estilizado) as demo:
        gr.HTML("<div class='custom-header'>Clasificador y Descripción de Imágenes Profesional</div>")
        gr.Markdown("### Sube una imagen para clasificarla y generar una descripción en español.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Selecciona o arrastra una imagen", type="pil")
                classify_button = gr.Button("🔍 Procesar", elem_classes="custom-button")
            with gr.Column():
                output_image = gr.Image(label="Imagen Original")
                output_label = gr.Textbox(label="Predicción (Clasificación)")
                output_desc = gr.Textbox(label="Descripción Generada")
        
        # Mensaje de estado para feedback al usuario
        progress_text = gr.Textbox(label="Estado", value="Esperando imagen...", interactive=False)
        
        def process_image(image):
            progress_text.value = "Procesando imagen..."
            outputs = classify_image(image, clf_processor, clf_model, caption_processor, caption_model, device)
            progress_text.value = "Procesamiento completado."
            return outputs
        
        classify_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[output_image, output_label, output_desc]
        )
    demo.launch()

if __name__ == "__main__":
    main()
