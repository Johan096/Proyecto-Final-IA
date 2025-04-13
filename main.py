## Nombre: Johan Daniel Muñoz salas 
## Matricula: 22-sisn-2-041
## Proyecto: Clasificador de imagenes 


import logging
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import gradio as gr
from PIL import Image
import openai
import os
import time
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Configura tu API key usando la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mensajes iniciales para GPT
mensajes = []
system_msg = (
    "Eres un experto en visión por computadora. Describe en español lo que probablemente se ve "
    "en una imagen, basándote en etiquetas de clasificación. Usa un lenguaje simple, útil y educativo."
)
mensajes.append({"role": "system", "content": system_msg})

def initialize_model(device: str):
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224").to(device)
    model.eval()
    return processor, model

def LLM(contexto):
    try:
        mensajes.append({"role": "user", "content": contexto})
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # O usa "gpt-4" si tienes acceso
            messages=mensajes
        )
        return respuesta["choices"][0]["message"]["content"]
    except Exception as e:
        time.sleep(5)
        print(f"Error de conexión: {e}")
        return LLM(contexto)

def classify_image(image, processor, model, device):
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        label = model.config.id2label[top_idx[0].item()]
        confidence = top_prob[0].item() * 100

        # Traducciones simples para algunas etiquetas
        traducciones = {
            "cat": "gato",
            "dog": "perro",
            "person": "persona",
            "car": "auto",
            "bicycle": "bicicleta"
        }
        label_es = traducciones.get(label.lower(), label)
        resultado = f"{label_es.capitalize()}: {confidence:.2f}%"

        contexto = (
            f"La imagen fue clasificada como '{label_es}' con una confianza del {confidence:.2f}%. "
            "Describe brevemente qué se observa en la imagen en español."
        )
        descripcion = LLM(contexto)
        return image, resultado, descripcion
    except Exception as e:
        logger.error(f"Error al clasificar: {e}")
        return None, "Error clasificando la imagen.", ""

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = initialize_model(device)
    
    # Interfaz personalizada usando Gradio con estilos CSS inyectados
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
        gr.HTML("<div class='custom-header'>Clasificador de Imágenes Profesional</div>")
        gr.Markdown("### Sube una imagen para clasificarla y obtener una descripción detallada.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Selecciona o arrastra una imagen", type="pil")
                classify_button = gr.Button("🔍 Clasificar", elem_classes="custom-button")
            with gr.Column():
                output_image = gr.Image(label="Imagen Original")
                output_label = gr.Textbox(label="Predicción")
                output_desc = gr.Textbox(label="Descripción en Español")
        
        # Mensaje de estado para dar feedback al usuario
        progress_text = gr.Textbox(label="Estado", value="Esperando imagen...", interactive=False)
        
        # Función para incluir feedback de procesamiento
        def process_image(image):
            progress_text.value = "Procesando imagen..."
            outputs = classify_image(image, processor, model, device)
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
