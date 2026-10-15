# CNNs vs Vision Transformers para la detección de melanoma (ISIC 2020)

Comparación de **ResNet-50**, **DenseNet-121** y **ViT-B/16** (preentrenados en ImageNet) bajo el mismo protocolo, con evaluación de desempeño, costo computacional, convergencia, ensambles e interpretabilidad (Grad-CAM y attention rollout).

| Modelo | AUC | F1 | Min/época |
|---|---|---|---|
| ResNet-50 | 0.890 | 0.219 | 3.8 |
| DenseNet-121 | 0.922 | 0.231 | 5.2 |
| ViT-B/16 | 0.920 | **0.329** | 10.9 |
| Ensamble 3 | **0.928** | 0.252 | — |

## Instalación

Python 3.14 y GPU NVIDIA. Instala PyTorch con CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/) y después:

```bash
pip install -r requirements.txt
```

## Datos y modelos

1. Descarga de [ISIC 2020](https://challenge.isic-archive.com/data/#2020) `ISIC_2020_Training_JPEG.zip`, `ISIC_2020_Training_GroundTruth_v2.csv` e `ISIC_2020_Training_Duplicates.csv` en `C:\datasets\isic2020\` (o cambia `DATASET_PATH` en los notebooks).
2. *(Opcional, evita reentrenar ~4 h)* Descarga los modelos del [Release v1.0](../../releases/tag/v1.0) o si ya estan descargados muevelos a `saved_models/`.

## Ejecución

Ejecuta los notebooks en orden:

| Notebook | Contenido |
|---|---|
| `00_preprocesamiento_imagenes` | Elimina duplicados y redimensiona a 256 px |
| `01_EDA_y_split` | Análisis exploratorio y partición 70/15/15 por paciente |
| `02_entrenamiento` | Entrena los 3 modelos (si existen en `saved_models/`, los carga) |
| `03_evaluacion_gradcam` | Métricas, bootstrap, ensambles, Grad-CAM y análisis crítico |
| `04_prediccion` | Predicción con los modelos guardados |

`isic_utils.py` contiene el `Dataset` y las métricas. 
