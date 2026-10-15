"""
Utilidades compartidas del proyecto final (ISIC 2020): dataset y métricas.

El Dataset vive en un archivo .py (y no dentro del notebook) porque en Windows
los workers del DataLoader (num_workers > 0) se crean como procesos nuevos que
necesitan importar la clase desde un módulo.
"""
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_auc_score)


## Datos

class ISICDataset(data.Dataset):

    def __init__(self, df, transform=None):
        """
        Inputs:
            df - DataFrame con las columnas "image_path" y "target"
            transform - Transformaciones de torchvision a aplicar a cada imagen
        """
        self.paths = df.image_path.tolist()
        self.targets = torch.tensor(df.target.values, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


## Métricas

def best_f1_threshold(y_true, y_prob):
    """Umbral que maximiza el F1 de la clase positiva (se elige en validación)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = np.argmax(f1[:-1])  # el último punto de la curva no tiene umbral
    return float(thresholds[best]), float(f1[best])


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Métricas de clasificación binaria (positivo = melanoma)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both_classes else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
