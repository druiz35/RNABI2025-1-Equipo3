import os
from classifiers import DriverClassifier

def test_on_folder(test_dir):
    model = DriverClassifier()
    total = 0
    correct = 0
    for class_name in os.listdir(test_dir):
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            # Solo procesa archivos de imagen
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue
            pred = model.predict(img_path)
            total += 1
            if pred == class_name:
                correct += 1
            else:
                print(f"Mal clasificada: {img_path} -> {pred} (debería ser {class_name})")
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(__file__), "test")
    test_on_folder(test_dir) 