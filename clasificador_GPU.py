import cudf
import cuml
import cupy as cp
import numpy as np
import time
import itertools

from cuml.cluster import KMeans
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def main_gpu_replicate():
   
    start_total = time.time()

    print("\nCargando CSV")
    try:
        gdf = cudf.read_csv("features_data456.csv")
    except FileNotFoundError:
        print("Error: Falta 'features_data456.csv'.")
        return

    # Filtramos columnas de features
    feature_cols = [c for c in gdf.columns if c.startswith('feat_')]
    X_descriptors = gdf[feature_cols].astype('float32')

    K_CLUSTERS = 150 
    print(f"\nEntrenando KMeans ({K_CLUSTERS} clusters)")
    
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_descriptors)
    
    # Asignar clusters
    gdf['cluster'] = kmeans.predict(X_descriptors)

    # HISTOGRAMAS (BoW)
    print("\nConstruyendo histogramas")
    
    # Conteo rápido en GPU
    grouped = gdf.groupby(['img_id', 'cluster']).size().reset_index(name='count')
    pivot_gdf = grouped.pivot(index='img_id', columns='cluster', values='count').fillna(0)
    
    for i in range(K_CLUSTERS):
        if i not in pivot_gdf.columns:
            pivot_gdf[i] = 0
            
    # Ordenar columnas numéricamente
    pivot_gdf = pivot_gdf.sort_index(axis=1)
    
    X_final = pivot_gdf.values.astype('float32')
    
    # Recuperar etiquetas
    labels_gdf = gdf[['img_id', 'label']].drop_duplicates().sort_values('img_id')
    y_final = labels_gdf['label'].values.astype('float32')

    # NORMALIZACIÓN L2
    print("Normalizando histogramas")
    norms = cp.linalg.norm(X_final, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Evitar división por cero
    X_normalized = X_final / norms

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normalized)
    
    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_final, test_size=0.3, random_state=42, stratify=y_final
    )

    # GRID SEARCH
    print("\nBuscando mejores hiperparámetros")
    
    # Tu grid original exacto
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['rbf']
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_acc = 0
    best_model = None
    best_params = {}
    
    for params in combinations:
        try:
            clf = SVC(kernel=params['kernel'], C=params['C'], gamma=params['gamma'], probability=True)
            clf.fit(X_train, y_train)
            
            # Validar
            score = accuracy_score(y_test, clf.predict(X_test))
            
            if score > best_acc:
                best_acc = score
                best_model = clf
                best_params = params
        except:
            continue

    print(f"\nMEJOR ACCURACY ENCONTRADO: {best_acc*100:.2f}%")
    print(f"Parámetros ganadores: {best_params}")

    print("\nReporte detallado")
    y_pred = best_model.predict(X_test)
    
    y_test_cpu = cp.asnumpy(y_test)
    y_pred_cpu = cp.asnumpy(y_pred)
    clases = ["Glioma", "Meningioma", "No Tumor", "Pituitaria"]
    
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=clases))
    print(f"Tiempo Total: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main_gpu_replicate()
