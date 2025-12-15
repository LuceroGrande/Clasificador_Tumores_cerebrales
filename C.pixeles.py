import cudf
import cuml
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cuml.decomposition import PCA
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def to_cpu(data):
    if hasattr(data, "to_numpy"):
        # Es un objeto cuDF (DataFrame o Series)
        return data.to_numpy()
    elif hasattr(data, "get"):
        # Es un objeto CuPy
        return data.get()
    elif isinstance(data, np.ndarray):
        # Ya es numpy
        return data
    else:
        # Fallback 
        return np.array(data)


def plot_cm(y_true, y_pred, target_names):
    # Aseguramos que entren datos de CPU
    y_true = to_cpu(y_true)
    y_pred = to_cpu(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicción')
    plt.show()

def plot_roc_multiclass(y_test, y_score, n_classes, target_names):
    # Aseguramos que entren datos de CPU
    y_test = to_cpu(y_test)
    y_score = to_cpu(y_score)
    
    # Binarizar etiquetas para One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {target_names[i]} (AUC = {roc_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC (Multiclase)')
    plt.legend(loc="lower right")
    plt.show()

def plot_learning_curve_custom(model, X, y):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    val_scores = []
    
    print("\nGenerando curva de aprendizaje")
    
    # Hacemos split inicial para tener un set de validación fijo
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for frac in train_sizes:
        # Tomar subconjunto de train
        size = int(len(X_train_full) * frac)
        # Slicing en objetos GPU (funciona en cudf y cupy)
        X_subset = X_train_full[:size]
        y_subset = y_train_full[:size]
        
        model.fit(X_subset, y_subset)
        
        # Predecir y evaluar (todo en GPU)
        pred_train = model.predict(X_subset)
        pred_val = model.predict(X_val)
        
        acc_train = accuracy_score(y_subset, pred_train)
        acc_val = accuracy_score(y_val, pred_val)
        
        train_scores.append(acc_train)
        val_scores.append(acc_val)
        
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * 100, train_scores, 'o-', color="r", label="Score Entrenamiento")
    plt.plot(train_sizes * 100, val_scores, 'o-', color="g", label="Score Validación")
    plt.title("Curva de Aprendizaje (SVM)")
    plt.xlabel("Porcentaje de datos de entrenamiento usados")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def main_gpu():
    print("Clasificación con pixeles en crudo")

    # Cargar Datos
    try:
        gdf = cudf.read_csv("pixel_data.csv")
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        return

    #  Preprocesamiento
    y = gdf['label'].astype('float32') # Esto es una Serie cudf
    pixel_cols = [c for c in gdf.columns if c.startswith('p')]
    X = gdf[pixel_cols].astype('float32')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print("Ejecutando PCA")
    pca = PCA(n_components=150)
    X_pca = pca.fit_transform(X_scaled) # Esto retorna un CuPy Array

    var = cp.sum(pca.explained_variance_ratio_)
    print(f"Varianza explicada: {var:.2f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenamiento
    print("Entrenando SVM")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # 6. Predicciones y Métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nACCURACY FINAL: {acc * 100:.2f}%")

    #  VISUALIZACIÓN (Conversión a CPU)
    
    # Usamos la función helper to_cpu para evitar el TypeError
    y_test_cpu = to_cpu(y_test)
    y_pred_cpu = to_cpu(y_pred)
    
    target_names = ["Glioma", "Meningioma", "No", "Pituitary"]

    print("\nREPORTE DE CLASIFICACIÓN")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=target_names))

    print("\nGenerando Matriz de Confusión")
    plot_cm(y_test_cpu, y_pred_cpu, target_names)

    print("\nGenerando Curvas ROC")
    y_proba = model.predict_proba(X_test)
    y_proba_cpu = to_cpu(y_proba) # Convertir probabilidades a CPU
    plot_roc_multiclass(y_test_cpu, y_proba_cpu, n_classes=4, target_names=target_names)

    # Para la curva de aprendizaje, pasamos los objetos de GPU originales ya que la función maneja el split y entrenamiento internamente en GPU
    plot_learning_curve_custom(model, X_pca, y)

if __name__ == "__main__":
    main_gpu()
