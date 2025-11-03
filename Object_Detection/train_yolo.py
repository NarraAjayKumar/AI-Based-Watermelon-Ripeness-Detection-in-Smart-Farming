import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def prepare_yolo_dataset():
    """Prepares train/val split and copies images and labels into YOLOv8 structure."""

    # Set paths
    image_dir = r'C:\MurthyLab\Object_Detection\dataset\images'
    label_dir = r'C:\MurthyLab\Object_Detection\dataset\labels'

    # Safety check
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"❌ Folder not found: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"❌ Folder not found: {label_dir}")

    # List images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_files) == 0:
        raise ValueError("⚠️ No image files found in 'dataset/images'. Add your data first.")

    # Split dataset (80% train, 20% val)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # Create YOLO folder structure
    for split in ['train', 'val']:
        os.makedirs(f'{split}/images', exist_ok=True)
        os.makedirs(f'{split}/labels', exist_ok=True)

    # Copy images and labels
    def copy_files(files, split):
        for file in files:
            shutil.copy2(os.path.join(image_dir, file), f'{split}/images/{file}')
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, f'{split}/labels/{label_file}')
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    print(f"✅ Dataset prepared: {len(train_files)} training and {len(val_files)} validation images.")

def train_yolo_model():
    """Trains YOLOv8 model using the prepared dataset and exports to ONNX."""

    # Load class names
    class_file = 'dataset/classes.txt'
    if not os.path.exists(class_file):
        raise FileNotFoundError(f"❌ Could not find class list at {class_file}")

    with open(class_file) as f:
        classes = [line.strip() for line in f if line.strip()]

    # Create data.yaml
    data_config = {
        'train': os.path.abspath('train/images'),
        'val': os.path.abspath('val/images'),
        'nc': len(classes),
        'names': classes
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    # Load and train model
    model = YOLO('yolov8s.pt')  # Change to yolov8n.pt or yolov8m.pt as needed
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='my_yolo_model',
        verbose=True
    )

    # Optional: Evaluate
    model.val(data='data.yaml')

    # Export to ONNX
    model.export(format='onnx')
    print("\n✅ Training complete! Model exported as ONNX.")

    return model

if __name__ == '__main__':
    print("🚀 Starting pipeline...")
    prepare_yolo_dataset()
    model = train_yolo_model()