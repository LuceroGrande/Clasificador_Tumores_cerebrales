import os
import cv2
import numpy as np
import scipy.io
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor

OUTPUT_CSV = "pixel_data.csv"
IMG_SIZE = (64, 64) # 4096 columnas por imagen
LABEL_MAP = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
MAT_TRANS = {1: 1, 2: 0, 3: 3} 

def process_image(args):
    path, kind, lbl = args
    img = None
    final_label = -1
    
    try:
        if kind == 'std':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            final_label = lbl
        elif kind == 'mat':
            try:
                mat = scipy.io.loadmat(path)
                l = int(mat['cjdata']['label'][0][0])
                raw = mat['cjdata']['image'][0][0]
            except:
                with h5py.File(path, 'r') as f:
                    l = int(f['cjdata']['label'][0][0])
                    raw = np.array(f['cjdata']['image'])
            
            if l not in MAT_TRANS: return None
            final_label = MAT_TRANS[l]
            raw = np.array(raw, dtype=np.float32)
            img = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img is not None:
            if len(img.shape) > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            # Convierte la matriz 64x64 en una fila de 4096 números
            return np.hstack(([final_label], img.flatten()))
            
    except:
        return None
    return None

def main():
    print("GENERANDO DATASET DE PÍXELES")
    tasks = []
    
    # Carpetas PNG
    if os.path.exists("./dataset"):
        folder_map = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}
        for split in ["Training", "Testing"]:
            p = os.path.join("./dataset", split)
            if os.path.exists(p):
                for f, l in folder_map.items():
                    fp = os.path.join(p, f)
                    if os.path.exists(fp):
                        for file in os.listdir(fp):
                            if file.endswith(('.png','.jpg')):
                                tasks.append((os.path.join(fp, file), 'std', l))

    # Carpetas MAT
    mats = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("brainTumorDataPublic")]
    for d in mats:
        for f in os.listdir(d):
            if f.endswith('.mat'):
                tasks.append((os.path.join(d, f), 'mat', None))

    print(f"Procesando {len(tasks)} imágenes...")
    
    with ProcessPoolExecutor() as exe:
        results = [r for r in exe.map(process_image, tasks) if r is not None]
    
    print(f"Guardando {len(results)} registros")
    cols = ['label'] + [f'p{i}' for i in range(IMG_SIZE[0]*IMG_SIZE[1])]
    pd.DataFrame(results, columns=cols).to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
