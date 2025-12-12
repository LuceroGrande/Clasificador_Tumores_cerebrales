import os
import numpy as np
import h5py
import cv2
from skimage.measure import marching_cubes
from tensorflow.keras.models import load_model


MODEL_PATH = "classifier_tumor.h5" 

print(f"Cargando modelo desde: {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

LABELS_MAP = {
    0: "Normal",
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}

def load_mat(filepath):
    try:
        with h5py.File(filepath, "r") as mat:
            cjdata = mat["cjdata"]
            image = np.array(cjdata["image"])
            tumorMask = np.array(cjdata["tumorMask"])
            return image, tumorMask
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return None, None

def preprocesar_imagen(img):
    # 1. Normalizar
    img_norm = (img - img.min()) / (img.max() - img.min())
    
    # 2. Redimensionar
    img_resized = cv2.resize(img_norm, (128, 128)) 
    
    # 3. Dar formato de batch (1, Alto, Ancho, Canales)
    img_final = np.expand_dims(img_resized, axis=-1)    
    return np.expand_dims(img_final, axis=0)

def run_pipeline(carpeta_raiz):
    print("Procesando carpeta:", carpeta_raiz)
    
    archivos = []
    for root, _, files in os.walk(carpeta_raiz):
        for f in files:
            if f.lower().endswith(".mat"):
                archivos.append(os.path.join(root, f))

    if not archivos:
        raise Exception("No hay archivos .mat en la carpeta")

    volumen_slices = []
    masks_slices = []
    predicciones = []

    for archivo in sorted(archivos):
        img, mask = load_mat(archivo)
        if img is None: continue
        
        volumen_slices.append(img)
        masks_slices.append(mask)

        # PREDICCIÓN
        if model:
            img_prep = preprocesar_imagen(img)
            pred = model.predict(img_prep, verbose=0)
            clase = np.argmax(pred)
            predicciones.append(clase)
        else:
            predicciones.append(0)

    if not volumen_slices:
        raise Exception("Error al cargar imágenes")

    # Unificar tamaños para visualización 3D
    target_shape = volumen_slices[0].shape
    if len(set(s.shape for s in volumen_slices)) > 1:
        volumen_slices = [cv2.resize(im, target_shape[::-1]) for im in volumen_slices]
        masks_slices = [cv2.resize(mk, target_shape[::-1]) for mk in masks_slices]

    volume = np.stack(volumen_slices, axis=0)
    mask_volume = np.stack(masks_slices, axis=0)

    # Calcular resultado final (Votación mayoritaria)
    total = len(predicciones)
    if total > 0:
        confidences = {name: predicciones.count(k)/total for k, name in LABELS_MAP.items()}
        idx_ganador = max(set(predicciones), key=predicciones.count)
        label_final = LABELS_MAP.get(idx_ganador, "Desconocido")
    else:
        label_final = "Error"
        confidences = {}

    # Generar malla 3D
    try:
        verts, faces, _, _ = marching_cubes(mask_volume, level=0.5)
    except:
        verts, faces = [], []

    return volume, mask_volume, verts, faces, label_final, confidences