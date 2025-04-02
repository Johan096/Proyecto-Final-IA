import logging
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configuración básica de logging para ver mensajes informativos en la terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model(device: str):
    """
    Inicializa y retorna el processor y modelo para clasificación de imágenes usando ViT.
    
    Args:
        device (str): Dispositivo a utilizar ('cuda' o 'cpu').
    
    Returns:
        processor: Processor para preprocesar la imagen.
        model: Modelo de clasificación movido al dispositivo indicado y en modo evaluación.
    """
    logger.info("Cargando el processor y el modelo para clasificación de imágenes...")
    try:
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        # Habilitamos output_attentions para la visualización interpretativa
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_attentions=True).to(device)
        model.eval()  # Modo evaluación para la inferencia
        logger.info("Modelo de clasificación cargado correctamente.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise e
    return processor, model

def compute_attention_rollout(attentions, discard_ratio=0.9):
    """
    Calcula la atención acumulada (rollout) a partir de las matrices de atención.
    
    Args:
        attentions (list[torch.Tensor]): Lista de tensores de atención de cada capa.
        discard_ratio (float): Proporción de atención que se descarta.
    
    Returns:
        torch.Tensor: Matriz de atención acumulada.
    """
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for att in attentions:
        att_heads = att.mean(dim=1)  # Promediar sobre las cabezas de atención
        # Sumar la identidad para incluir la conexión residual
        att_heads = att_heads + torch.eye(att_heads.size(-1)).to(att_heads.device)
        att_heads = att_heads / att_heads.sum(dim=-1, keepdim=True)
        result = torch.matmul(att_heads, result)
    return result

def overlay_heatmap_on_image(image, heatmap):
    """
    Superpone un heatmap sobre la imagen original para visualizar áreas de atención.
    
    Args:
        image (PIL.Image.Image): Imagen original.
        heatmap (PIL.Image.Image): Imagen del heatmap en escala de grises.
    
    Returns:
        PIL.Image.Image: Imagen combinada con el heatmap superpuesto.
    """
    heatmap_np = np.array(heatmap)
    cmap = plt.get_cmap('jet')
    heatmap_color = cmap(heatmap_np/255.0)  # Valores RGBA entre 0 y 1
    heatmap_color = np.uint8(heatmap_color*255)
    heatmap_color = Image.fromarray(heatmap_color).convert("RGBA")
    image_rgba = image.convert("RGBA")
    blended = Image.blend(image_rgba, heatmap_color, alpha=0.5)
    return blended

def classify_image(image: Image.Image, processor, model, device: str, top_k: int = 1, 
                   show_attention: bool = False, use_tta: bool = False, show_probability_chart: bool = False):
    """
    Clasifica la imagen y opcionalmente genera un heatmap de atención y un gráfico de barras de probabilidades.
    
    Args:
        image (PIL.Image.Image): Imagen de entrada.
        processor: Processor para preprocesar la imagen.
        model: Modelo de clasificación.
        device (str): Dispositivo de inferencia.
        top_k (int): Número de predicciones principales a mostrar.
        show_attention (bool): Si True, genera y retorna el heatmap de atención.
        use_tta (bool): Si True, utiliza Test Time Augmentation (flip horizontal) para robustez.
        show_probability_chart (bool): Si True, muestra un gráfico de barras con las probabilidades de las predicciones.
    
    Returns:
        tuple: (str: resultados de clasificación, PIL.Image.Image or None: heatmap de atención,
                PIL.Image.Image or None: gráfico de probabilidades)
    """
    logger.info("Clasificando la imagen...")
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        # Aplicar Test Time Augmentation (TTA) si está activado
        if use_tta:
            inputs_orig = processor(images=image, return_tensors="pt")
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            inputs_flip = processor(images=flipped, return_tensors="pt")
            inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
            inputs_flip = {k: v.to(device) for k, v in inputs_flip.items()}
            with torch.no_grad():
                outputs_orig = model(**inputs_orig, output_attentions=show_attention)
                outputs_flip = model(**inputs_flip, output_attentions=show_attention)
            logits = (outputs_orig.logits + outputs_flip.logits) / 2
            # Para la visualización de atención se usa la salida de la imagen original
            attn_output = outputs_orig.attentions if show_attention else None
        else:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=show_attention)
            logits = outputs.logits
            attn_output = outputs.attentions if show_attention else None

        # Calcular probabilidades y obtener las top_k predicciones
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = model.config.id2label[idx.item()]
            results.append(f"{label}: {prob.item()*100:.2f}%")
        result_text = "\n".join(results)

        # Generar gráfico de barras de las probabilidades si se solicita
        prob_chart_image = None
        if show_probability_chart:
            fig, ax = plt.subplots(figsize=(6,4))
            labels = [model.config.id2label[idx.item()] for idx in top_indices[0]]
            probabilities = [prob.item() for prob in top_probs[0]]
            ax.bar(labels, probabilities, color='skyblue')
            ax.set_ylim(0,1)
            ax.set_ylabel('Probabilidad')
            ax.set_title('Probabilidades de las predicciones')
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            prob_chart_image = Image.open(buf)
            plt.close(fig)

        # Generar el heatmap de atención si se solicita
        attn_image = None
        if show_attention and attn_output is not None:
            rollout = compute_attention_rollout(attn_output)
            # Extraer el mapa de atención del token de clase (excluyendo el token [CLS])
            attn_map = rollout[0, 0, 1:]
            num_patches = attn_map.shape[0]
            grid_size = int(num_patches ** 0.5)
            attn_map = attn_map.reshape(grid_size, grid_size).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)  # Normalizar a [0,1]
            attn_map_img = Image.fromarray((attn_map*255).astype("uint8")).resize(image.size, resample=Image.BILINEAR)
            attn_image = overlay_heatmap_on_image(image, attn_map_img)

        return result_text, attn_image, prob_chart_image
    except Exception as e:
        logger.error(f"Error al clasificar la imagen: {e}")
        return "Error clasificando la imagen. Verifica la imagen y los parámetros.", None, None

def main():
    """
    Función principal que configura y lanza la interfaz de Gradio para el Clasificador de Imágenes Profesional.
    Incorpora características de robustez, eficiencia e interpretabilidad.
    """
    # Configurar el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilizando dispositivo: {device}")

    # Inicializar el processor y el modelo
    processor, model = initialize_model(device)

    # CSS personalizado para una presentación profesional y dinámica
    css = """
    .gradio-container {
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
    }
    .gradio-title {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .gradio-description, .gradio-article {
        font-size: 1.1em;
        color: #333;
    }
    """

    # Configurar la interfaz de Gradio con opciones para top_k, TTA, atención y gráfico de probabilidades
    interface = gr.Interface(
        fn=lambda image, top_k, show_attention, use_tta, show_probability_chart: classify_image(
            image, processor, model, device, top_k, show_attention, use_tta, show_probability_chart
        ),
        inputs=[
            gr.Image(type="pil", label="Sube una imagen para clasificar"),
            gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Número de predicciones (Top K)"),
            gr.Checkbox(label="Mostrar mapa de atención", value=False),
            gr.Checkbox(label="Usar Test Time Augmentation (flip horizontal)", value=False),
            gr.Checkbox(label="Mostrar gráfico de probabilidades", value=False)
        ],
        outputs=[
            gr.Textbox(label="Predicciones"),
            gr.Image(label="Mapa de Atención (opcional)"),
            gr.Image(label="Gráfico de Probabilidades (opcional)")
        ],
        title="Clasificador de Imágenes Profesional",
        description=(
            "Bienvenido al Clasificador de Imágenes Profesional. "
            "Esta aplicación utiliza un modelo de deep learning basado en ViT para clasificar imágenes. "
            "Cuenta con técnicas de robustez como TTA y ofrece visualizaciones interpretativas, como mapas de atención y gráficos de probabilidades."
        ),
        article=(
            "Desarrollado con PyTorch y Hugging Face, este clasificador ofrece una interfaz profesional y dinámica "
            "mediante Gradio. Se han incorporado mejoras que permiten obtener predicciones más claras y una mejor interpretación de los resultados."
        ),
        css=css,
        allow_flagging="never"
    )

    # Lanzar la interfaz en un servidor local
    interface.launch()

if __name__ == "__main__":
    main()
