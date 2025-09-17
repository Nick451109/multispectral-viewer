import gradio as gr
import tifffile
import numpy as np
import os
from matplotlib import pyplot as plt

# Directorio con imágenes TIFF
IMAGE_DIR = "ruta"

# Obtener lista de imágenes
def get_tiff_files():
    return [f for f in os.listdir(IMAGE_DIR) if f.endswith(".tiff") or f.endswith(".tif")]

# Función para visualizar
def view_tiff(filename, channels):
    path = os.path.join(IMAGE_DIR, filename)
    img = tifffile.imread(path)  # (H, W, C)
    
    # Seleccionar canales
    selected = []
    for ch in channels:
        idx = int(ch.split("_")[-1])  # "Canal_0", "Canal_1", etc.
        if idx < img.shape[-1]:
            selected.append(img[..., idx])
    
    if not selected:
        return None
    
    # Si es más de un canal -> stack en RGB-like
    if len(selected) == 3:
        out = np.stack(selected, axis=-1)
    else:
        out = selected[0]  # un solo canal
    
    # Normalizar a [0, 255] para visualización
    out = (out - out.min()) / (out.max() - out.min() + 1e-6) * 255
    out = out.astype(np.uint8)
    
    return out

def update_channels(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = tifffile.imread(path)
    num_channels = img.shape[-1]
    return gr.CheckboxGroup.update(choices=[f"Canal_{i}" for i in range(num_channels)])

with gr.Blocks() as demo:
    with gr.Row():
        file_dropdown = gr.Dropdown(choices=get_tiff_files(), label="Selecciona imagen")
        channels = gr.CheckboxGroup(choices=[], label="Canales")
    output = gr.Image(type="numpy")
    
    file_dropdown.change(update_channels, inputs=file_dropdown, outputs=channels)
    channels.change(view_tiff, inputs=[file_dropdown, channels], outputs=output)

demo.launch()
