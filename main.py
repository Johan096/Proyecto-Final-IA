import logging
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import gradio as gr
from PIL import Image
from openai import OpenAI
import os
import time

# Inicializa cliente de OpenAI (usa variable de entorno o escribe tu clave)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mensajes para GPT
mensajes = []
system_msg = "Eres un experto en visión por computadora. Describe en español lo que probablemente se ve en una imagen, basándote en etiquetas de clasificación. Usa un lenguaje simple, útil y educativo."
mensajes.append({"role": "system", "content": system_msg})

def initialize_model(device: str):
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224").to(device)
    model.eval()
    return processor, model

def LLM(contexto):
    try:
        mensajes.append({"role": "user", "content": contexto})
        respuesta = client.chat.completions.create(
            model="gpt-4o",  # puedes cambiar a gpt-3.5-turbo si lo prefieres
            messages=mensajes
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        time.sleep(5)
        print(f"Error de conexión: {e}")
        return LLM(contexto)

def classify_image(image, processor, model, device):
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)

        label = model.config.id2label[top_idx[0].item()]
        confidence = top_prob[0].item() * 100

        # Traducción básica
        traducciones = {
            "cat": "gato", "dog": "perro", "person": "persona", "car": "auto", "bicycle": "bicicleta"
        }
        label_es = traducciones.get(label.lower(), label)

        resultado = f"{label_es.capitalize()}: {confidence:.2f}%"

        # Pedir descripción a GPT
        contexto = f"La imagen fue clasificada como '{label_es}' con una confianza del {confidence:.2f}%. Describe qué hay en la imagen."
        descripcion = LLM(contexto)

        return image, resultado, descripcion

    except Exception as e:
        logger.error(f"Error al clasificar: {e}")
        return None, "Error clasificando la imagen.", ""

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = initialize_model(device)

    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ Clasificador de Imágenes (ViT + GPT en Español)")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="📷 Sube una imagen", type="pil")
                classify_button = gr.Button("🔍 Clasificar")
            with gr.Column():
                output_image = gr.Image(label="Imagen Original")
                output_label = gr.Textbox(label="Predicción")
                output_desc = gr.Textbox(label="Descripción en Español")

        classify_button.click(
            fn=lambda img: classify_image(img, processor, model, device),
            inputs=[image_input],
            outputs=[output_image, output_label, output_desc]
        )

    demo.launch()

if __name__ == "__main__":
    main()
