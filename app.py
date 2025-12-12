from flask import Flask, render_template, request, jsonify
from pipeline import run_pipeline

import os
import base64
import matplotlib
matplotlib.use("Agg")  # Evitar problemas de Tkinter
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

app = Flask(__name__)

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

@app.route("/")
def home():
    return render_template("pagina.html")

@app.route("/procesar", methods=["POST"])
def procesar():
    archivos = request.files.getlist("folder")

    if not archivos:
        return jsonify({"error": "No se recibieron archivos"}), 400

    # Crear carpeta temporal
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)

    # Limpiar carpeta previa
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            os.remove(os.path.join(root, f))

    # Guardar archivos
    for file in archivos:
        ruta = os.path.join(temp_dir, file.filename)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        file.save(ruta)

    volume, mask_volume, verts, faces, label, confidences = run_pipeline(temp_dir)

    # Slice 2D: superposición de segmentación
    slices_b64 = []
    for i in range(volume.shape[0]):
        slice_i = volume[i]
        mask_i = mask_volume[i]

        # Normalizar slice
        img2d = ((slice_i - slice_i.min()) / (slice_i.max() - slice_i.min()) * 255).astype(np.uint8)

        # Imagen RGB
        img_rgb = np.stack([img2d]*3, axis=-1)

        # Superponer máscara en rojo con transparencia
        alpha = 0.5
        img_rgb[mask_i > 0] = (img_rgb[mask_i > 0] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)

        # Convertir a base64
        fig = plt.figure(figsize=(4,4))
        plt.imshow(img_rgb)
        plt.axis("off")
        slices_b64.append(fig_to_base64(fig))
        plt.close()

    # Render 3D
    fig3 = plt.figure(figsize=(4,4))
    ax = fig3.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    img3d_b64 = fig_to_base64(fig3)
    plt.close()

    return jsonify({
        "label": label,
        "img2d_slices": slices_b64,
        "img3d": img3d_b64,
        "confidences": confidences
    })

if __name__ == "__main__":
    app.run(debug=True)
