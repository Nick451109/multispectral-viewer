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

def resolve_path(base_dir, filename):
    """Devuelve la ruta absoluta manejando dropdown (str) y upload (list)."""
    if not filename:
        return None
    if isinstance(filename, list):  # si viene de File upload
        if len(filename) == 0:
            return None
        return filename[0]
    return os.path.join(base_dir, filename)  # si viene de dropdown

def update_channels_from_path(path):
    """Actualiza dinámicamente los canales según la imagen"""
    if not path:
        return gr.update(choices=[], value=[])
    if isinstance(path, list):
        path = path[0]
    img = tifffile.imread(path)
    num_channels = img.shape[-1] if img.ndim == 3 else 1
    return gr.update(
        choices=[f"Canal_{i}" for i in range(num_channels)],
        value=[f"Canal_{i}" for i in range(min(num_channels, 4))]
    )

def view_tiff_from_path(path, channels):
    """Carga y visualiza una imagen TIFF desde una ruta completa"""
    if not path:
        return None
    if isinstance(path, list):
        path = path[0]

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

        if idx == 0:
            canvas[..., 0] += data
        elif idx == 1:
            canvas[..., 1] += data
        elif idx == 2:
            canvas[..., 2] += data
        else:
            thermal = cm.inferno(data)[..., :3]
            canvas = np.maximum(canvas, thermal)

    canvas = np.clip(canvas, 0, 1)
    return (canvas * 255).astype(np.uint8)

with gr.Blocks() as demo:
    with gr.Row():
        file_upload = gr.File(label="O sube/arrastra un archivo TIFF", type="filepath")

    with gr.Row():
        folder_text = gr.Textbox(label="Ruta de carpeta", placeholder="/ruta/a/carpeta")
        scan_btn = gr.Button("Escanear carpeta")
    with gr.Row():
        file_dropdown = gr.Dropdown(choices=[], label="Selecciona imagen desde carpeta")

    channels = gr.CheckboxGroup(choices=[], label="Canales", interactive=True)
    output = gr.Image(type="numpy", label="Vista TIFF")#, height=600, width=800)

    # Dropdown flow
    scan_btn.click(lambda d: gr.update(choices=get_tiff_files(d)), inputs=folder_text, outputs=file_dropdown)
    file_dropdown.change(lambda d, f: update_channels_from_path(resolve_path(d, f)),
                         inputs=[folder_text, file_dropdown], outputs=channels)
    file_dropdown.change(lambda d, f, c: view_tiff_from_path(resolve_path(d, f), c),
                         inputs=[folder_text, file_dropdown, channels], outputs=output)
    channels.change(lambda d, f, c: view_tiff_from_path(resolve_path(d, f), c),
                    inputs=[folder_text, file_dropdown, channels], outputs=output)

    # Upload flow
    file_upload.change(update_channels_from_path, inputs=file_upload, outputs=channels)
    file_upload.change(view_tiff_from_path, inputs=[file_upload, channels], outputs=output)
    channels.change(view_tiff_from_path, inputs=[file_upload, channels], outputs=output)

demo.launch(server_name="127.0.0.1", server_port=7860)
