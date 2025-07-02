#!/usr/bin/env python3
"""
Script de configuración para el Sistema de Clasificación de Conducción Distractiva
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detectado")
    return True

def create_virtual_environment():
    """Crea un entorno virtual"""
    if os.path.exists("venv"):
        print("✅ Entorno virtual ya existe")
        return True
    
    print("🔧 Creando entorno virtual...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Entorno virtual creado exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al crear el entorno virtual")
        return False

def install_requirements():
    """Instala las dependencias del proyecto"""
    print("📦 Instalando dependencias...")
    
    # Determinar el comando de pip según el sistema operativo
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar dependencias")
        return False

def check_model_file():
    """Verifica que el archivo del modelo esté disponible"""
    model_path = "notebooks/modulo2/Resnet18.pth"
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado en: {model_path}")
        return True
    else:
        print(f"⚠️  Advertencia: No se encontró el modelo en {model_path}")
        print("   Asegúrate de tener el archivo Resnet18.pth antes de ejecutar la aplicación")
        return False

def check_example_images():
    """Verifica que las imágenes de ejemplo estén disponibles"""
    example_dir = "notebooks/modulo2"
    example_images = [f"image_set_{i}.png" for i in range(5)]
    
    found_images = []
    for img in example_images:
        if os.path.exists(os.path.join(example_dir, img)):
            found_images.append(img)
    
    if found_images:
        print(f"✅ {len(found_images)} imágenes de ejemplo encontradas")
        return True
    else:
        print("⚠️  No se encontraron imágenes de ejemplo")
        print("   La funcionalidad de imágenes de ejemplo no estará disponible")
        return False

def main():
    """Función principal del script de configuración"""
    print("🚗 Configurando Sistema de Clasificación de Conducción Distractiva")
    print("=" * 60)
    
    # Verificar versión de Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear entorno virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_requirements():
        sys.exit(1)
    
    # Verificar archivos necesarios
    check_model_file()
    check_example_images()
    
    print("\n" + "=" * 60)
    print("✅ Configuración completada")
    print("\n🚀 Para ejecutar la aplicación:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
        print("   streamlit run app.py")
    else:
        print("   source venv/bin/activate")
        print("   streamlit run app.py")
    
    print("\n📖 Para más información, consulta el README.md")

if __name__ == "__main__":
    main() 