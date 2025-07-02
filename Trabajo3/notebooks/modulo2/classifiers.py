import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import kagglehub

from sklearn.metrics import roc_auc_score, confusion_matrix

import seaborn as sns
###### from sklearn.metrics import roc_auc_score
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
    RESTNET_PATH = "./Resnet18.pth"
    def __init__(self): 
        self.class_names = [
            "other_actvities",
            "safe_driving",
            "talking_phone",
            "texting_phone",
            "turning"
        ]
        self.set_transform_pipeline()
        self.set_resnet()

    def set_resnet(self): 
        self.model = models.resnet18()
        self.model.load_state_dict(torch.load(DriverClassifier.RESTNET_PATH, map_location=torch.device('cpu')))
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
        self.model.eval()

    def set_custom_model(self): ... # TODO: Implementar modelo custom 

    def set_transform_pipeline(self): 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            print(predicted_idx)
        return self.class_names[predicted_idx]


if __name__ == "__main__":
    model = DriverClassifier()
    image_path = "./safe_test.png"
    predicted_class = model.predict(image_path)
    print(predicted_class)