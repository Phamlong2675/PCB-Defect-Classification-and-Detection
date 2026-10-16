import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Configuration
DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/resnet18")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16
EPOCHS = 20
FREEZE_EPOCHS = 3
PATIENCE = 8
LR_HEAD = 1e-3
LR_BASE = 1e-4
SEED = 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load data
manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = manifest.drop_duplicates("class_index").sort_values("class_index")["class_name"].tolist()
num_classes = len(class_names)

train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_rows = manifest[manifest["split"] == "valid"].reset_index(drop=True)

# Data preprocessing
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=18,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def read_batch(rows, indices, transform):
    batch = rows.iloc[indices.tolist()]
    images = []
    for path in batch["crop_path"]:
        with Image.open(DATA_DIR / path) as image:
            images.append(transform(image.convert("RGB")))
    labels = torch.tensor(batch["class_index"].to_numpy(), dtype=torch.long)
    return torch.stack(images).to(device), labels.to(device)


train_loader = DataLoader(range(len(train_rows)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(range(len(val_rows)), batch_size=BATCH_SIZE)

# Build ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes))
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

# Stage 1: Freeze backbone
model.requires_grad_(False)
model.fc.requires_grad_(True)

optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD)

# Stage 2: Fine-tune all layers
backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]

optimizer_fine = torch.optim.AdamW([
    {"params": backbone, "lr": LR_BASE},
    {"params": model.fc.parameters(), "lr": LR_HEAD}])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fine, T_max=(EPOCHS - FREEZE_EPOCHS) * len(train_loader))

# Evaluation
@torch.no_grad()
def evaluate(loader, rows):
    model.eval()
    total_loss = 0
    predictions = []

    for indices in loader:
        images, labels = read_batch(rows, indices, test_transform)
        logits = model(images)

        total_loss += loss_fn(logits, labels).item() * len(labels)
        predictions.extend(logits.argmax(1).cpu().numpy())

    y_true = rows["class_index"].to_numpy()
    predictions = np.array(predictions)

    val_loss = total_loss / len(rows)
    accuracy = np.mean(predictions == y_true)
    macro_f1 = f1_score(y_true, predictions, labels=np.arange(num_classes), average="macro", zero_division=0)
    return val_loss, accuracy, macro_f1, predictions

# Training
best_f1, best_epoch, wait = -1.0, 0, 0
for epoch in range(1, EPOCHS + 1):
    fine_tune = epoch > FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS + 1:
        model.requires_grad_(True)

    model.train(fine_tune)
    if not fine_tune:
        model.fc.train()

    optimizer = optimizer_fine if fine_tune else optimizer_head
    for indices in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        images, labels = read_batch(train_rows, indices, train_transform)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        if fine_tune:
            scheduler.step()

    # Validation
    val_f1 = evaluate(val_loader, val_rows)[2]
    print(f"Epoch {epoch}/{EPOCHS} | Val Macro-F1: {val_f1:.4f}")

    # Early Stopping
    if val_f1 > best_f1:
        best_f1, best_epoch, wait = val_f1, epoch, 0
        best_weights = copy.deepcopy(model.state_dict())
    elif fine_tune:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping!")
            break

# Restore and save best model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), OUT_DIR / "resnet18_model.pth")