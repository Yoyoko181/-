#!e:\毕设\Terra-main\model_train\.venv\Scripts\python.exe
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path

# === Configuration ===
DATA_DIR = r'e:\毕设\Terra-main\model_train\cwt_images'
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_PRETRAINED = False

class FootstepDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _predict(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(predicted.detach().cpu().numpy().astype(int).tolist())
    acc = correct / total if total else 0.0
    return y_true, y_pred, float(acc)


def _save_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    cmx = confusion_matrix(y_true, y_pred, labels=labels)
    denom = cmx.sum(axis=1, keepdims=True)
    denom = (denom == 0) + denom
    cmn = cmx.astype("float32") / denom.astype("float32")
    plt.figure(figsize=(10, 8))
    plt.imshow(cmn, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    if len(labels) <= 30:
        ticks = list(range(len(labels)))
        tick_labels = [str(x) for x in labels]
        plt.xticks(ticks, tick_labels, rotation=90, fontsize=6)
        plt.yticks(ticks, tick_labels, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()

def train_model():
    # 1. Prepare Data
    all_image_paths = []
    all_labels = []
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
    num_classes = len(classes)
    
    for cls_name in classes:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        img_paths = glob.glob(os.path.join(cls_dir, '*.png'))
        all_image_paths.extend(img_paths)
        all_labels.extend([class_to_idx[cls_name]] * len(img_paths))
    
    print(f"Total images found: {len(all_image_paths)}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {DEVICE}")
    label_list = sorted(set(all_labels))

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FootstepDataset(train_paths, train_labels, transform=transform)
    val_dataset = FootstepDataset(val_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Build Model (ResNet18)
    weights = ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_footstep_model.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    y_true_val, y_pred_val, val_acc_final = _predict(model, val_loader, DEVICE)
    cm_path = Path("confusion_matrix_val.png")
    _save_confusion_matrix(y_true_val, y_pred_val, label_list, cm_path)

    metrics = {
        "task": "train_cnn",
        "device": str(DEVICE),
        "data_dir": str(DATA_DIR),
        "num_classes": int(num_classes),
        "num_train_images": int(len(train_dataset)),
        "num_val_images": int(len(val_dataset)),
        "epochs": int(NUM_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "lr": float(LEARNING_RATE),
        "use_pretrained": bool(USE_PRETRAINED),
        "best_val_acc": float(best_acc),
        "val_acc_final": float(val_acc_final),
        "best_model_path": str(Path("best_footstep_model.pth").resolve()),
        "training_history_path": str(Path("training_history.png").resolve()),
        "confusion_matrix_val_path": str(cm_path.resolve()),
    }
    metrics_path = Path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {cm_path}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.axhline(val_acc_final, linestyle="--", linewidth=1.5, label=f"Val Acc: {val_acc_final:.4f}")
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

if __name__ == '__main__':
    train_model()
