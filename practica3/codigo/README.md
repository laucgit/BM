# Generación y validación de datos sintéticos sobre un dataset médico tabular

Este proyecto usa el dataset **Breast Cancer Wisconsin (Diagnostic)** y entrena una estrategia de generación sintética basada en **Gaussian Mixture Models (GMM) condicionales por clase**.

## Estructura

- `src/train_and_evaluate.py`: entrena el generador, crea el conjunto sintético y calcula métricas de fidelidad, utilidad y privacidad.
- `outputs/`: resultados generados automáticamente.
- `outputs/figures/`: figuras empleadas en el informe.

## Dataset público

- UCI Machine Learning Repository: `https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic`

## Librería empleada

- scikit-learn (GaussianMixture): `https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html`

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta `codigo/`:

```bash
python src/train_and_evaluate.py
```

El script:

1. carga el dataset médico tabular,
2. divide en train/test real,
3. ajusta un GMM por clase con selección de componentes mediante BIC,
4. genera un conjunto sintético del mismo tamaño que el train real,
5. valida fidelidad, utilidad y privacidad,
6. guarda tablas, figuras y resúmenes en `outputs/`.

## Notas metodológicas

- **Fidelidad**: diferencias de media y desviación típica, KS, Wasserstein y diferencia entre matrices de correlación.
- **Utilidad**: comparación **TRTR** (Train Real - Test Real) frente a **TSTR** (Train Synthetic - Test Real) con regresión logística y random forest.
- **Privacidad**: DCR, NNDR y tasa de duplicados exactos respecto al train real.
