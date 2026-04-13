# Entrega: generación y validación de datos sintéticos

## Estructura
- `codigo/train_and_evaluate.py`: script principal para entrenar el generador y evaluar fidelidad, utilidad y privacidad.
- `codigo/outputs/`: métricas, dataset sintético y figuras generadas automáticamente.
- `informe_datos_sinteticos_final.tex`: fuente LaTeX del informe.
- `informe_datos_sinteticos_final.pdf`: informe final compilado.

## Requisitos
Instalar las dependencias listadas en `requirements.txt`.

## Ejecución
Desde la carpeta `codigo/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train_and_evaluate.py --cv-folds 5
```

Esto genera:
- métricas por variable y por fold,
- resultados de utilidad TRTR/TSTR,
- métricas de privacidad y membership inference,
- figuras en `codigo/outputs/figures/`,
- `summary_metrics.json` con el resumen global.

## Compilación del informe
Desde la carpeta raíz de la entrega:

```bash
pdflatex -interaction=nonstopmode -halt-on-error informe_datos_sinteticos_final.tex
pdflatex -interaction=nonstopmode -halt-on-error informe_datos_sinteticos_final.tex
```

## Dataset
Breast Cancer Wisconsin (Diagnostic), repositorio UCI.
