import cudf
import cuml
import cupy as cp
import time
from cuml.decomposition import PCA
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from sklearn.metrics import classification_report

def main_gpu():
    print("CLASIFICACIÓN con pixeles en crudo")

    print("Cargando CSV")
    gdf = cudf.read_csv("pixel_data.csv")

    # Separar
    y = gdf['label'].astype('float32')
    pixel_cols = [c for c in gdf.columns if c.startswith('p')]
    X = gdf[pixel_cols].astype('float32')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print("Ejecutando PCA")
    pca = PCA(n_components=150)
    X_pca = pca.fit_transform(X_scaled)

    var = cp.sum(pca.explained_variance_ratio_)
    print(f"Varianza explicada: {var:.2f}") # Debería ser > 0.90

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando SVM")
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nACCURACY FINAL: {acc * 100:.2f}%")

    y_pred_cpu = cp.asnumpy(model.predict(X_test))
    y_test_cpu = cp.asnumpy(y_test)
    print(classification_report(y_test_cpu, y_pred_cpu,
                                target_names=["Glioma", "Meningioma", "No", "Pituitary"]))

if __name__ == "__main__":
    main_gpu()
