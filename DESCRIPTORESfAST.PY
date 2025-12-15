import os
import cv2
import numpy as np
import scipy.io
import pandas as pd
import time
import h5py
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


#dataset con caracteristicas obtenidas
PATH_CARPETAS_CLASIFICADAS = "./dataset" 
OUTPUT_CSV = "features_data456.csv"
IMG_SIZE = (128, 128)
LABEL_MAP_FINAL = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
MAT_TRANSLATION = {1: 1, 2: 0, 3: 3} 

def get_descriptors_single_image(args):
    path, source_type, label_info, img_global_id_start = args
    
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
    
        if img is not None:
            if len(img.shape) > 2: 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            
            keypoints = fast.detect(img, None)
            if not keypoints: return None
            
            keypoints, descriptors = brief.compute(img, keypoints)
            if descriptors is not None and len(descriptors) > 0:
                return (descriptors, final_label)
                
    except Exception:
        return None
    return None

def run_extraction(num_cores, guardar=False):
    start_time = time.time()
    tasks = []
    
    # Recolección de archivos (PNGs)
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

    # Recolección de archivos (MATs)
    carpetas_mat = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("brainTumorDataPublic")]
    for carpeta in carpetas_mat:
        full_path = os.path.join('.', carpeta)
        if os.path.exists(full_path):
            archivos_mat = [f for f in os.listdir(full_path) if f.endswith('.mat')]
            for file in archivos_mat:
                tasks.append((os.path.join(full_path, file), 'mat', None, 0))

    total_files = len(tasks)
    if total_files == 0:
        return 0

    buffer_data = []
    img_global_id = 0
    
    print(f"Ejecutando con {num_cores} núcleos (Archivos: {total_files})...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(get_descriptors_single_image, tasks)
        
        for res in results:
            if not guardar:
                pass # Solo medimos tiempo
            elif res is not None:
                desc, label = res
                n_desc = desc.shape[0]
                meta = np.column_stack((np.full(n_desc, img_global_id), np.full(n_desc, label)))
                buffer_data.extend(np.hstack((meta, desc)))
                img_global_id += 1

    if guardar and buffer_data:
        print(f"   -> Guardando {len(buffer_data)} descriptores en {OUTPUT_CSV}...")
        cols = ['img_id', 'label'] + [f'feat_{j}' for j in range(32)]
        df = pd.DataFrame(buffer_data, columns=cols)
        df.to_csv(OUTPUT_CSV, index=False)
        print("   -> Archivo guardado.")

    end_time = time.time()
    return end_time - start_time

def run_benchmark():
    max_cpu = multiprocessing.cpu_count()
    # 1, 2, 4, ... hasta max_cpu
    core_counts = [1]
    curr = 2
    while curr <= max_cpu:
        core_counts.append(curr)
        curr *= 2
    if core_counts[-1] != max_cpu:
        core_counts.append(max_cpu)
    
    times = []
    
    print(f"INICIANDO BENCHMARK (Max CPU: {max_cpu}) ")
    
    # Obtener Tiempos
    for c in core_counts:
        es_ultimo = (c == core_counts[-1])
        
        t = run_extraction(c, guardar=es_ultimo)
        times.append(t)
        print(f"   Core(s): {c} | Tiempo: {t:.4f} s")

    # Calcular Métricas
    t1 = times[0] # Tiempo secuencial 
    speedups = [t1 / t for t in times]
    efficiencies = [s / c for s, c in zip(speedups, core_counts)]
    
    # Graficar
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Tiempo de Ejecución
    ax1.plot(core_counts, times, 'o-', color='blue')
    ax1.set_title('Tiempo de Ejecución vs Núcleos')
    ax1.set_xlabel('Número de Procesadores')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.grid(True)
    
    # Speedup
    ax2.plot(core_counts, speedups, 'o-', color='green', label='Real')
    ax2.plot(core_counts, core_counts, '--', color='gray', label='Ideal (Lineal)') # Speedup ideal
    ax2.set_title('Speedup')
    ax2.set_xlabel('Número de Procesadores')
    ax2.set_ylabel('Speedup (T1 / Tp)')
    ax2.legend()
    ax2.grid(True)
    
    # Eficiencia
    ax3.plot(core_counts, efficiencies, 'o-', color='red')
    ax3.set_title('Eficiencia')
    ax3.set_xlabel('Número de Procesadores')
    ax3.set_ylabel('Eficiencia (Speedup / p)')
    ax3.set_ylim(0, 1.1) # suele estar entre 0 y 1
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_resultados.png')
    print("\nGráficas guardadas en 'benchmark_resultados.png'")
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    run_benchmark()
