# 🧬 Predicción de Espectros de Dicroísmo Circular mediante PINNs

Este repositorio contiene el código fuente del Trabajo de Fin de Grado (TFG) enfocado en la predicción de espectros de Dicroísmo Circular (CD) Electrónico para derivados de helicenos utilizando Deep Learning y Redes Neuronales Informadas por la Física (PINNs).

## 🚀 Arquitectura del Proyecto

El proyecto ha sido diseñado siguiendo los estándares de **Cookiecutter Data Science** e **Ingeniería de Software Científico**, garantizando modularidad, reproducibilidad y escalabilidad.

```text
proyecto_helicenos/
├── data/
│   ├── raw/               <- Datos originales (Omitidos en GitHub por peso)
│   ├── processed/         <- Tensores y parámetros extraídos (Listos para ML)
├── models/                <- Redes neuronales entrenadas (.pt)
├── notebooks/             <- Cuadernos Jupyter de exploración y validación
├── reports/
│   └── figures/           <- Gráficos vectoriales generados por la evaluación
├── src/                   <- Código fuente de producción
│   ├── data/              <- Dataloaders, extracción cuántica y GMM
│   ├── models/            <- Arquitecturas de PyTorch y Funciones de Pérdida
│   ├── train/             <- Orquestador de Optuna, K-Fold y Entrenamiento
│   └── evaluate/          <- Generación de gráficos y métricas de dominio
├── main.py                <- Orquestador Maestro (Panel de Control)
└── requirements.txt       <- Dependencias del proyecto