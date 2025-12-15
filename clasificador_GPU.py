import cudf
import cuml
import cupy as cp
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from cuml.cluster import KMeans
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def main_gpu_replicate():

    start_total = time.time()

    print("\nCargando CSV")
    try:
        gdf = cudf.read_csv("features_data456.csv")
    except FileNotFoundError:
        print("Error: Falta el archivo CSV.")
        return

    # Filtramos columnas de features
    feature_cols = [c for c in gdf.columns if c.startswith('feat_')]
    X_descriptors = gdf[feature_cols].astype('float32')

    K_CLUSTERS = 1000
    print(f"\nEntrenando KMeans")

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_descriptors)

    # Asignar clusters
    gdf['cluster'] = kmeans.predict(X_descriptors)

    # HISTOGRAMAS (BoW)
    print("\nConstruyendo histogramas")

    grouped = gdf.groupby(['img_id', 'cluster']).size().reset_index(name='count')
    pivot_gdf = grouped.pivot(index='img_id', columns='cluster', values='count').fillna(0)

    for i in range(K_CLUSTERS):
        if i not in pivot_gdf.columns:
            pivot_gdf[i] = 0

    pivot_gdf = pivot_gdf.sort_index(axis=1)

    X_final = pivot_gdf.values.astype('float32')

    # Recuperar etiquetas
    labels_gdf = gdf[['img_id', 'label']].drop_duplicates().sort_values('img_id')
    y_final = labels_gdf['label'].values.astype('float32')

    # NORMALIZACIÓN
    print("Normalizando histogramas")
    X_final = cp.sqrt(X_final)

    norms = cp.linalg.norm(X_final, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_normalized = X_final / norms

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normalized)

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_final, test_size=0.3, random_state=42, stratify=y_final
    )

    # GRID SEARCH
    print("\nBuscando mejores hiperparámetros")

    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': ['scale', 0.01, 0.001],
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
            score = accuracy_score(y_test, clf.predict(X_test))

            if score > best_acc:
                best_acc = score
                best_model = clf
                best_params = params
        except:
            continue

    print(f"\nMEJOR ACCURACY ENCONTRADO: {best_acc*100:.2f}%")
    print(f"Parámetros ganadores: {best_params}")

    print("\nGenerando Curvas de Aprendizaje")
    train_sizes = np.linspace(0.2, 1.0, 4)
    train_scores = []
    val_scores = []
    n_train_total = X_train.shape[0]

    for fraction in train_sizes:
        subset_size = int(n_train_total * fraction)
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]

        model_lc = SVC(
            kernel=best_params['kernel'],
            C=best_params['C'],
            gamma=best_params['gamma'],
            probability=True
        )

        try:
            model_lc.fit(X_subset, y_subset)
            train_acc = accuracy_score(y_subset, model_lc.predict(X_subset))
            val_acc = accuracy_score(y_test, model_lc.predict(X_test))

            train_scores.append(train_acc)
            val_scores.append(val_acc)
            print(f"  Subset {fraction*100:.0f}% -> Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        except Exception as e:
            print(f"  Saltando punto {fraction}: {e}")

    if len(train_scores) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes[:len(train_scores)] * n_train_total, train_scores, 'o-', color="r", label="Entrenamiento")
        plt.plot(train_sizes[:len(train_scores)] * n_train_total, val_scores, 'o-', color="g", label="Validación")
        plt.title(f"Curva de Aprendizaje (Accuracy: {best_acc*100:.2f}%)")
        plt.xlabel("Ejemplos")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    y_pred = best_model.predict(X_test)
    y_test_cpu = cp.asnumpy(y_test)
    y_pred_cpu = cp.asnumpy(y_pred)
    clases = ["Glioma", "Meningioma", "No Tumor", "Pituitaria"]

    print("\nGenerando Matriz de Confusión")
    cm = confusion_matrix(y_test_cpu, y_pred_cpu)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.show()

    print("\nGenerando Curvas ROC")

    # Obtener probabilidades
    y_score_gpu = best_model.predict_proba(X_test)
    y_score = cp.asnumpy(y_score_gpu)

    # Binarizar etiquetas
    y_test_bin = label_binarize(y_test_cpu, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {clases[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC Multiclase')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print("\nReporte Final")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=clases))
    print(f"Tiempo Total: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main_gpu_replicate()
