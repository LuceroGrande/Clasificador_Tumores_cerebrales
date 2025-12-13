import os
import cv2
import numpy as np
import scipy.io
import pandas as pd
import time
import h5py
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

PATH_CARPETAS_CLASIFICADAS = "./dataset" 
OUTPUT_CSV = "features_data456.csv"
IMG_SIZE = (128, 128)

# Mapeos
LABEL_MAP_FINAL = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
MAT_TRANSLATION = {1: 1, 2: 0, 3: 3} 

def get_descriptors_single_image(args):
    path, source_type, label_info, img_global_id_start = args
    
    # Inicializar FAST y BRIEF
    try:
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    except AttributeError:
        return None

    img = None
    final_label = -1
    
    try:
        if source_type == 'standard':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            final_label = label_info 
            
        elif source_type == 'mat':
            try:
                mat_data = scipy.io.loadmat(path)
                label_mat = int(mat_data['cjdata']['label'][0][0])
                img_raw = mat_data['cjdata']['image'][0][0]
            except:
                with h5py.File(path, 'r') as f:
                    label_mat = int(f['cjdata']['label'][0][0])
                    img_raw = np.array(f['cjdata']['image'])
            
            if label_mat not in MAT_TRANSLATION: return None 
            final_label = MAT_TRANSLATION[label_mat]
            
            img_raw = np.array(img_raw, dtype=np.float32)
            img = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
        # PROCESAMIENTO Y EXTRACCIÓN
        if img is not None:
            if len(img.shape) > 2: 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            img = cv2.resize(img, IMG_SIZE)
            
            # Detectar Keypoints con FAST
            keypoints = fast.detect(img, None)
            
            if not keypoints:
                return None
            
            # Calcular Descriptores con BRIEF
            keypoints, descriptors = brief.compute(img, keypoints)
            
            if descriptors is not None and len(descriptors) > 0:
                return (descriptors, final_label)
                
    except Exception as e:
        return None
        
    return None

def main_parallel():
    start_time = time.time()
    print(f"Iniciando Extracción con FAST + BRIEF")
    
    tasks = []
    
    # PNGs
    folder_to_id = {
        "glioma_tumor": LABEL_MAP_FINAL["glioma"],
        "meningioma_tumor": LABEL_MAP_FINAL["meningioma"],
        "no_tumor": LABEL_MAP_FINAL["no_tumor"],
        "pituitary_tumor": LABEL_MAP_FINAL["pituitary"]
    }
    
    if os.path.exists(PATH_CARPETAS_CLASIFICADAS):
        for split in ["Training", "Testing"]:
            p_split = os.path.join(PATH_CARPETAS_CLASIFICADAS, split)
            if not os.path.exists(p_split): continue
            for fname, lbl in folder_to_id.items():
                fpath = os.path.join(p_split, fname)
                if not os.path.exists(fpath): continue
                for file in os.listdir(fpath):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        tasks.append((os.path.join(fpath, file), 'standard', lbl, 0))

    # MATs
    carpetas_mat = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("brainTumorDataPublic")]
    print(f"   Carpetas .mat detectadas: {len(carpetas_mat)}")

    for carpeta in carpetas_mat:
        full_path = os.path.join('.', carpeta)
        archivos_mat = [f for f in os.listdir(full_path) if f.endswith('.mat')]
        print(f"   Añadiendo carpeta {carpeta}: {len(archivos_mat)} archivos")
        for file in archivos_mat:
            tasks.append((os.path.join(full_path, file), 'mat', None, 0))

    total_files = len(tasks)
    print(f"Total de archivos a procesar: {total_files}")
    
    if total_files == 0: return

    buffer_data = []
    img_global_id = 0
    chunk_size = 10000 
    first_write = True
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    print(f"Procesando")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(get_descriptors_single_image, tasks)
        
        for i, res in enumerate(results):
            if res is not None:
                desc, label = res
                n_desc = desc.shape[0]
                
                meta = np.column_stack((np.full(n_desc, img_global_id), np.full(n_desc, label)))
                buffer_data.extend(np.hstack((meta, desc)))
                
                img_global_id += 1
            
            if i % 500 == 0: print(f"   Progreso: {i}/{total_files}...", end='\r')

            if len(buffer_data) >= chunk_size:
                cols = ['img_id', 'label'] + [f'feat_{j}' for j in range(32)]
                df = pd.DataFrame(buffer_data, columns=cols)
                df.to_csv(OUTPUT_CSV, mode='w' if first_write else 'a', header=first_write, index=False)
                buffer_data = []
                first_write = False

    if buffer_data:
        cols = ['img_id', 'label'] + [f'feat_{j}' for j in range(32)]
        df = pd.DataFrame(buffer_data, columns=cols)
        df.to_csv(OUTPUT_CSV, mode='w' if first_write else 'a', header=first_write, index=False)

    print(f"\nTotal imágenes exitosas: {img_global_id}")
    print(f"Archivo guardado: {OUTPUT_CSV}")
    print(f"Tiempo total: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main_parallel()