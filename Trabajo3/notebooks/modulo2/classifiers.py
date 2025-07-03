import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

import kagglehub

from sklearn.metrics import roc_auc_score, confusion_matrix

import seaborn as sns
import time
import numpy as np
from PIL import Image

# EJEMPLO DE USO
"""
model = DriverClassifier()
image_path = "path/to/image.jpg"
predicted_class = model.predict(image_path)
print(predicted_class)
"""

class DriverClassifier:
    # Ruta del modelo ResNet18 preentrenado guardado localmente
    RESTNET_PATH = os.path.join(os.path.dirname(__file__), "Resnet18.pth")

    def __init__(self): 
        # Clases que el modelo puede predecir (5 clases de actividad al volante)
        self.class_names = [
            "other_actvities",
            "safe_driving",
            "talking_phone",
            "texting_phone",
            "turning"
        ]
        # Configuramos la transformación de imagen (resize, tensor, normalización)
        self.set_transform_pipeline()
        # Cargamos el modelo ResNet18 preentrenado con pesos guardados
        self.set_resnet()

    def set_resnet(self): 
        self.model = models.resnet18()
        self.model.load_state_dict(torch.load(DriverClassifier.RESTNET_PATH, map_location=torch.device('cpu')))
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)  # Cambia la capa final DESPUÉS
        self.model.eval()

    def set_custom_model(self): ...# Método placeholder para implementar un modelo custom si se desea

    def set_transform_pipeline(self): 
        # Definimos las transformaciones que se aplicarán a cada imagen antes de la predicción
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),          # Resize a tamaño compatible con ResNet18
            transforms.ToTensor(),                   # Convertir imagen a tensor PyTorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalización estándar imagenes RGB
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, show_preprocessed=False):
        # Abrimos la imagen y la convertimos a RGB
        image = Image.open(image_path).convert("RGB")
        # Aplicamos las transformaciones y agregamos dimensión batch
        input_tensor = self.transform(image).unsqueeze(0)
        if show_preprocessed:
            import matplotlib.pyplot as plt
            img_np = input_tensor.squeeze(0).permute(1,2,0).numpy()
            img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Desnormaliza
            img_np = np.clip(img_np, 0, 1)
            plt.imshow(img_np)
            plt.title('Imagen que recibe el modelo (224x224)')
            plt.axis('off')
            plt.show()
        with torch.no_grad():
            # Ejecutamos la inferencia
            output = self.model(input_tensor)
            # Obtenemos la clase con la probabilidad más alta
            predicted_idx = torch.argmax(output, dim=1).item()
        # Retornamos el nombre de la clase predicha
        return self.class_names[predicted_idx]


if __name__ == "__main__":
    # Instanciamos el clasificador
    model = DriverClassifier()
    # Ruta a la imagen que queremos clasificar
    image_path = "./safe_test.png"
    # Obtenemos la predicción para la imagen
    predicted_class = model.predict(image_path)
    # Imprimimos la clase predicha
    print(predicted_class)
