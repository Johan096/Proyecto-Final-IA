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
        processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
        # Habilitamos output_attentions para la visualización interpretativa
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224", output_attentions=True).to(device)
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

def classify_image(image: Image.Image, processor, model, device: str, 
                   top_k: int = 1, show_attention: bool = False, use_tta: bool = False, 
                   show_probability_chart: bool = False, idioma: str = "Español", 
                   modo_oscuro: bool = False):
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
        idioma (str): "Español" o "Inglés" para la presentación de etiquetas.
        modo_oscuro (bool): Si True, ajusta los gráficos al modo oscuro.
    
    Returns:
        tuple: (PIL.Image.Image: imagen original, str: resultados de clasificación, 
                PIL.Image.Image or None: heatmap de atención, PIL.Image.Image or None: gráfico de probabilidades)
    """
    logger.info("Clasificando la imagen...")
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        original_img = image.copy()  # Guardamos la imagen original para el preview

        # Diccionario de traducción de etiquetas de ejemplo
        translation_dict = {
            "Dog": "Perro",
            "Cat": "Gato",
            "dog": "perro",
            "cat": "gato"
        }

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
            if idioma == "Español":
                label = translation_dict.get(label, label)
            results.append(f"{label}: {prob.item()*100:.2f}%")
        result_text = "\n".join(results)

        # Generar gráfico de barras de las probabilidades si se solicita
        prob_chart_image = None
        if show_probability_chart:
            fig, ax = plt.subplots(figsize=(6,4))
            labels = [model.config.id2label[idx.item()] for idx in top_indices[0]]
            if idioma == "Español":
                labels = [translation_dict.get(label, label) for label in labels]
            probabilities = [prob.item() for prob in top_probs[0]]
            ax.bar(labels, probabilities, color='skyblue')
            ax.set_ylim(0,1)
            ax.set_ylabel('Probabilidad')
            ax.set_title('Probabilidades de las predicciones')
            if modo_oscuro:
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                ax.title.set_color('white')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
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

        return original_img, result_text, attn_image, prob_chart_image
    except Exception as e:
        logger.error(f"Error al clasificar la imagen: {e}")
        return None, "Error clasificando la imagen. Verifica la imagen y los parámetros.", None, None

def main():
    """
    Función principal que configura y lanza la interfaz de Gradio para el Clasificador de Imágenes Profesional.
    Incorpora mejoras visuales y de usabilidad sin alterar la funcionalidad original.
    """
    # Configurar el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilizando dispositivo: {device}")

    # Inicializar el processor y el modelo
    processor, model = initialize_model(device)

    # CSS personalizado para una presentación profesional y dinámica
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Clasificador de Imágenes Profesional")
        gr.Markdown("Esta aplicación utiliza un modelo ViT para clasificar imágenes y ofrece interpretabilidad a través de mapas de atención y gráficos de probabilidad. Además, cuenta con soporte multilenguaje y modo oscuro/claro.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Sube una imagen para clasificar", type="pil")
                idioma_dropdown = gr.Dropdown(choices=["Español", "Inglés"], value="Español", label="Idioma")
                modo_oscuro_checkbox = gr.Checkbox(label="Modo Oscuro", value=False)
                with gr.Accordion("Opciones Avanzadas", open=False):
                    top_k_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Número de predicciones (Top K)")
                    show_attention_checkbox = gr.Checkbox(label="Mostrar mapa de atención", value=False)
                    use_tta_checkbox = gr.Checkbox(label="Usar Test Time Augmentation (flip horizontal)", value=False)
                    show_prob_chart_checkbox = gr.Checkbox(label="Mostrar gráfico de probabilidades", value=False)
                classify_button = gr.Button("Clasificar")
            with gr.Column(scale=1):
                gr.Markdown("### Resultados")
                original_image_output = gr.Image(label="Imagen Original")
                classification_output = gr.Textbox(label="Predicciones")
                attention_output = gr.Image(label="Mapa de Atención (opcional)")
                prob_chart_output = gr.Image(label="Gráfico de Probabilidades (opcional)")

        # Al hacer clic en el botón se ejecuta la función de clasificación
        classify_button.click(
            fn=lambda image, top_k, show_attention, use_tta, show_prob_chart, idioma, modo_oscuro: 
                classify_image(image, processor, model, device, top_k, show_attention, use_tta, show_prob_chart, idioma, modo_oscuro),
            inputs=[image_input, top_k_slider, show_attention_checkbox, use_tta_checkbox, show_prob_chart_checkbox, idioma_dropdown, modo_oscuro_checkbox],
            outputs=[original_image_output, classification_output, attention_output, prob_chart_output]
        )
    
        gr.Markdown("Desarrollado con PyTorch, Hugging Face y Gradio.")

    demo.launch()

if __name__ == "__main__":
    main()
