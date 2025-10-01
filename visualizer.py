import gradio as gr
import tifffile
import numpy as np
import os, glob
import matplotlib.cm as cm

def get_tiff_files(image_dir):
    """Escanea recursivamente todos los .tiff dentro de image_dir"""
    if not image_dir or not os.path.isdir(image_dir):
        return []
    files = glob.glob(os.path.join(image_dir, "**", "*.tif"), recursive=True)
    files += glob.glob(os.path.join(image_dir, "**", "*.tiff"), recursive=True)
    return [os.path.relpath(f, image_dir) for f in files]

def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-6)

def view_tiff(image_dir, filename, channels):
    """Carga y visualiza una imagen TIFF con los canales seleccionados"""
    if not filename:
        return None

    path = os.path.join(image_dir, filename)
    img = tifffile.imread(path)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    h, w, c = img.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    for ch in channels:
        idx = int(ch.split("_")[-1])
        if idx >= c:
            continue

        data = normalize(img[..., idx])

        # Comportamiento RGB clásico
        if idx == 0:   # Red
            canvas[..., 0] += data
        elif idx == 1: # Green
            canvas[..., 1] += data
        elif idx == 2: # Blue
            canvas[..., 2] += data
        else:
            # Cualquier otro canal lo mostramos en "inferno" como overlay
            thermal = cm.inferno(data)[..., :3]
            canvas = np.maximum(canvas, thermal)

    canvas = np.clip(canvas, 0, 1)
    return (canvas * 255).astype(np.uint8)

def update_channels(image_dir, filename):
    """Actualiza dinámicamente la lista de canales"""
    if not filename:
        return gr.update(choices=[], value=[])
    path = os.path.join(image_dir, filename)
    img = tifffile.imread(path)
    num_channels = img.shape[-1] if img.ndim == 3 else 1
    return gr.update(
        choices=[f"Canal_{i}" for i in range(num_channels)],
        value=[f"Canal_{i}" for i in range(min(num_channels, 4))]  # por defecto RGB + IR
    )

with gr.Blocks() as demo:
    with gr.Row():
        folder_text = gr.Textbox(label="Ruta de carpeta", placeholder="/Volumes/T7/NICK/resultados/haze_left/aliked")
    with gr.Row():
        scan_btn = gr.Button("Escanear carpeta")
        file_dropdown = gr.Dropdown(choices=[], label="Selecciona imagen")
        channels = gr.CheckboxGroup(choices=[], label="Canales", interactive=True)
    output = gr.Image(type="numpy", label="Vista TIFF" )#, height=600, width=800)

    scan_btn.click(
        lambda d: gr.update(choices=get_tiff_files(d)),
        inputs=folder_text,
        outputs=file_dropdown
    )

    file_dropdown.change(update_channels, inputs=[folder_text, file_dropdown], outputs=channels)
    file_dropdown.change(view_tiff, inputs=[folder_text, file_dropdown, channels], outputs=output)
    channels.change(view_tiff, inputs=[folder_text, file_dropdown, channels], outputs=output)

demo.launch(server_name="127.0.0.1", server_port=7860)
