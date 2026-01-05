# Clasificador de tumores cerebrales (MRI): SVM (CPU/GPU) + aplicación web (Flask)

Este repositorio reúne scripts y utilidades para **clasificación de tumores cerebrales en imágenes de resonancia magnética (MRI/IRM)** y experimentos de **cómputo paralelo**.

El proyecto tiene **dos vertientes**:

1. **Aplicación web (Flask)** para procesar un conjunto de cortes `.mat`, generar visualización 2D/3D y devolver una **predicción** (requiere un modelo Keras `.h5`).
2. **Experimentos offline** para:
   - construir datasets a partir de píxeles crudos (`pixel_data.csv`) y entrenar un **SVM con PCA** en GPU (cuML),
   - extraer descriptores **FAST + BRIEF** en CPU (paralelizado) y entrenar un pipeline **BoVW + SVM** en GPU (cuML).

> Nota importante: por cómo está el repositorio actualmente, **no incluye** el modelo entrenado `classifier_tumor.h5`, la plantilla HTML `pagina.html` ni los datasets de imágenes. Aun así, los pasos para ejecutar están documentados abajo.

---

## Contenido del repositorio

| Archivo | Propósito |
|---|---|
| `app.py` | Servidor Flask. Recibe una carpeta (upload) con archivos `.mat` y responde con predicción + visualizaciones en base64. |
| `pipeline.py` | Carga un modelo Keras (`classifier_tumor.h5`), lee `.mat`, preprocesa y predice por corte. También genera una malla 3D con `marching_cubes`. |
| `E.Pixeles.py` | Construye un dataset basado en píxeles crudos (`pixel_data.csv`) desde imágenes PNG/JPG y/o `.mat` (paralelo CPU). |
| `C.pixeles.py` | Entrena/evalúa un pipeline en GPU: normalización + PCA + SVM (cuML) usando `pixel_data.csv`. |
| `DESCRIPTORESfAST.PY` | Extrae descriptores FAST+BRIEF en CPU, paraleliza con `ProcessPoolExecutor` y guarda `features_data456.csv`. |
| `clasificador_GPU.py` | Entrena en GPU: KMeans (BoVW) + SVM con búsqueda de hiperparámetros usando `features_data456.csv`. |
| `requirements.txt` | Dependencias base (ver notas: está incompleto para la app web y RAPIDS suele instalarse con conda). |

---

## Requisitos

### Requisitos mínimos (para ejecutar scripts CPU)
- Linux/macOS/Windows
- Python 3.9+ (recomendado 3.10/3.11)
- Compiladores/headers estándar (según tu sistema) para algunas dependencias

### Para ejecutar la aplicación web (Flask + Keras)
Además de lo anterior:
- **Flask**
- **TensorFlow** (Keras)
- **h5py**
- **scikit-image** (por `marching_cubes`)

### Para ejecutar los experimentos en GPU (cuML/cuDF/CuPy)
- GPU NVIDIA + drivers compatibles
- CUDA compatible con tu entorno
- Instalación de RAPIDS/cuML/cuDF (recomendado vía **conda/mamba**)

---

## Instalación (entorno virtual recomendado)

1) Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependencias base:

```bash
pip install -r requirements.txt
```

3) Instalar dependencias necesarias para la app web (no están en `requirements.txt`):

```bash
pip install flask tensorflow scikit-image h5py
```

> Si vas a usar FAST+BRIEF, necesitas el módulo `xfeatures2d`, que normalmente viene con:
>
> ```bash
> pip install opencv-contrib-python
> ```
>
> (En lugar de `opencv-python`).

---

## Datos esperados (datasets)

El repositorio está diseñado para trabajar con **dos tipos de entradas**:

### A) Archivos `.mat` (Figshare/Brain Tumor Dataset)
- Varios scripts buscan carpetas en el directorio del proyecto cuyo nombre empiece por:
  - `brainTumorDataPublic...`
- Dentro esperan archivos con extensión `.mat`.

### B) Imágenes PNG/JPG (estructura tipo “Brain Tumor Dataset”)
Para los scripts de píxeles y descriptores, si existe una carpeta `./dataset`, se espera la estructura:

```
dataset/
  Training/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/
  Testing/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/
```

---

## Ejecución 1: Aplicación web (Flask)

### Archivos necesarios
1) **Modelo Keras**
- Debe existir un archivo `classifier_tumor.h5` en la raíz del proyecto (el código lo carga desde esa ruta).

2) **Plantilla HTML**
- El servidor intenta renderizar `pagina.html`. Flask normalmente espera este archivo en `templates/pagina.html`.
- Actualmente el repositorio no incluye dicha carpeta/archivo, por lo que debes agregarlo.

### Ejecutar el servidor

```bash
python app.py
```

Por defecto corre en `http://127.0.0.1:5000`.

### ¿Qué hace al procesar?
- El endpoint `/procesar` recibe una lista de archivos (subida de carpeta) y guarda temporalmente en `temp_upload/`.
- Busca archivos `.mat`, lee `image` y `tumorMask`, predice por corte con el modelo, y:
  - genera overlays 2D de segmentación (máscara sobre el corte),
  - intenta construir una malla 3D a partir del volumen de máscaras.

> Si `classifier_tumor.h5` no existe o no se carga, el pipeline hace un fallback y devuelve una etiqueta “Normal” (clase 0) en la votación.

---

## Ejecución 2: Experimento basado en píxeles (PCA + SVM en GPU)

### Paso 1: Generar `pixel_data.csv`

```bash
python E.Pixeles.py
```

Salida: `pixel_data.csv` en la raíz del proyecto.

### Paso 2: Entrenar/evaluar en GPU

```bash
python C.pixeles.py
```

Este script:
- carga `pixel_data.csv` con cuDF,
- normaliza, aplica PCA (`n_components=150`) y entrena un SVM RBF (cuML),
- imprime métricas y grafica matriz de confusión y curvas ROC.

---

## Ejecución 3: FAST+BRIEF + BoVW + SVM (CPU + GPU)

### Paso 1: Extraer descriptores en paralelo (CPU)

```bash
python DESCRIPTORESfAST.PY
```

Salida: `features_data456.csv`.

**Nota:** si `cv2.xfeatures2d` no está disponible, el extractor BRIEF no se creará. Instala `opencv-contrib-python`.

### Paso 2: Entrenar BoVW + SVM en GPU

```bash
python clasificador_GPU.py
```

Este script:
- hace KMeans (BoVW) con `K=1000`,
- construye histogramas por imagen,
- normaliza y entrena un SVM con búsqueda de hiperparámetros,
- muestra métricas (matriz de confusión, ROC, etc.).

---

## Problemas comunes (troubleshooting)

### 1) “No se encuentra pagina.html” / Error de plantilla
- Crea `templates/pagina.html` (Flask busca templates en la carpeta `templates/`).

### 2) “Error al cargar el modelo classifier_tumor.h5”
- Asegúrate de tener el archivo `classifier_tumor.h5` en la raíz.
- Verifica que tu versión de TensorFlow sea compatible con el modelo.

### 3) `cv2.xfeatures2d` no existe
- Instala `opencv-contrib-python` y asegúrate de no tener conflictos con `opencv-python`.

### 4) Errores instalando `cudf/cuml/cupy`
- Normalmente RAPIDS se instala con conda/mamba, y depende fuertemente de la versión de CUDA.
- Si no cuentas con GPU, ejecuta únicamente la parte CPU (por ejemplo, generación de CSVs) o adapta los scripts a scikit-learn.

---

## Resultados (referencia)

En el informe original del proyecto se menciona aproximadamente:
- ~81% usando descriptores FAST+BRIEF
- ~93% usando píxeles crudos + PCA + SVM

Estos valores dependen del dataset exacto, partición, preprocesamiento y parámetros.
