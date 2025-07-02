#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento de la aplicación
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

def test_imports():
    """Prueba que todas las importaciones funcionen"""
    print("🔍 Probando importaciones...")
    
    try:
        import streamlit as st
        print("✅ Streamlit importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Streamlit: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando PyTorch: {e}")
        return False
    
    try:
        from torchvision import transforms, models
        print("✅ TorchVision importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando TorchVision: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Pillow: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Matplotlib: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Pandas: {e}")
        return False
    
    return True

def test_model_loading():
    """Prueba la carga del modelo"""
    print("\n🤖 Probando carga del modelo...")
    
    model_path = "notebooks/modulo2/Resnet18.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        return False
    
    try:
        # Crear modelo ResNet18
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 5)
        
        # Cargar pesos
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        print("✅ Modelo cargado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return False

def test_image_preprocessing():
    """Prueba el preprocesamiento de imágenes"""
    print("\n🖼️  Probando preprocesamiento de imágenes...")
    
    try:
        # Crear una imagen de prueba
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Transformaciones
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Aplicar transformaciones
        image_tensor = transform(test_image).unsqueeze(0)
        
        print(f"✅ Preprocesamiento exitoso - Shape: {image_tensor.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return False

def test_prediction():
    """Prueba la predicción con una imagen de ejemplo"""
    print("\n🎯 Probando predicción...")
    
    # Buscar una imagen de ejemplo
    example_images = [f"notebooks/modulo2/image_set_{i}.png" for i in range(5)]
    test_image_path = None
    
    for img_path in example_images:
        if os.path.exists(img_path):
            test_image_path = img_path
            break
    
    if test_image_path is None:
        print("⚠️  No se encontraron imágenes de ejemplo para la prueba")
        return True  # No es un error crítico
    
    try:
        # Cargar modelo
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 5)
        model.load_state_dict(torch.load("notebooks/modulo2/Resnet18.pth", map_location=torch.device('cpu')))
        model.eval()
        
        # Cargar y preprocesar imagen
        image = Image.open(test_image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Hacer predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        classes_es = ["Otras actividades", "Conducción segura", "Hablando por teléfono", "Enviando mensajes", "Girando"]
        
        print(f"✅ Predicción exitosa:")
        print(f"   Comportamiento: {classes_es[predicted_class]}")
        print(f"   Confianza: {confidence:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False

def test_streamlit_config():
    """Prueba la configuración de Streamlit"""
    print("\n⚙️  Probando configuración de Streamlit...")
    
    config_path = ".streamlit/config.toml"
    
    if os.path.exists(config_path):
        print("✅ Archivo de configuración de Streamlit encontrado")
        return True
    else:
        print("⚠️  Archivo de configuración de Streamlit no encontrado")
        return True  # No es crítico

def main():
    """Función principal de pruebas"""
    print("🧪 Ejecutando pruebas del Sistema de Clasificación")
    print("=" * 50)
    
    tests = [
        ("Importaciones", test_imports),
        ("Carga del modelo", test_model_loading),
        ("Preprocesamiento de imágenes", test_image_preprocessing),
        ("Predicción", test_prediction),
        ("Configuración Streamlit", test_streamlit_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! La aplicación está lista para ejecutarse.")
        print("\n🚀 Para ejecutar la aplicación:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores antes de ejecutar la aplicación.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 