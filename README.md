# Clasificación de Tumores Cerebrales con SVM y Cómputo Paralelo

Este repositorio contiene la implementación del proyecto **Clasificación de tumores cerebrales en imágenes de resonancia magnética (IRM)** utilizando **Máquinas de Vectores de Soporte (SVM)** y **técnicas de cómputo paralelo en CPU y GPU**, desarrollado para la materia de *Cómputo Paralelo*.

## Objetivo
Implementar un pipeline eficiente e interpretable para la clasificación de tumores cerebrales (glioma, meningioma, pituitario y cerebro sano), optimizando las etapas más costosas mediante paralelización.

## Metodología
1. Carga de imágenes MRI (Figshare y Brain Tumor Dataset).
2. Preprocesamiento de imágenes.
3. Extracción paralela de características FAST + BRIEF (CPU).
4. Construcción de Bag of Visual Words con K-Means (GPU).
5. Clasificación y ajuste de hiperparámetros con SVM (GPU).
6. Evaluación mediante métricas estándar.

Se incluye un experimento comparativo basado en píxeles crudos + PCA + SVM.

## Estructura del repositorio
brain-tumor-svm-parallel/
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
├── results/
└── docs/

## Ejecución
Pipeline principal:
python src/build_bovw_svm_gpu.py

## Resultados
Accuracy del pipeline principal: ~84%.
