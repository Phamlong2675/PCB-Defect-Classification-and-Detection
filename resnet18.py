import copy
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# Configuration
DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/resnet18")
BATCH_SIZE = 16
EPOCHS = 20
FREEZE_EPOCHS = 3
PATIENCE = 8
LR_HEAD = 1e-3
LR_BASE = 1e-4
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load data
manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = manifest.drop_duplicates("class_index").sort_values("class_index")["class_name"].tolist()
num_classes = len(class_names)
train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_rows = manifest[manifest["split"] == "valid"].reset_index(drop=True)
test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
print("Images per split:", manifest["split"].value_counts().to_dict())

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet mean and std
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.05, 0.05), scale=(0.9, 1.1),
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    normalize])
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

# The loaders only give row numbers; the images of a batch are read when the batch is used.
train_loader = DataLoader(range(len(train_rows)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(range(len(val_rows)), batch_size=BATCH_SIZE)
test_loader = DataLoader(range(len(test_rows)), batch_size=BATCH_SIZE)

def read_batch(rows, indices, transform):
    batch = rows.iloc[indices.tolist()]
    images = []
    for path in batch["crop_path"]:
        with Image.open(DATA_DIR / path) as image:
            images.append(transform(image.convert("L").convert("RGB")))   # gray crop -> 3 channels
    labels = torch.tensor(batch["class_index"].to_numpy())
    return torch.stack(images).to(device), labels.to(device)

# ResNet18 pretrained on ImageNet
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(512, num_classes))   # 512 features
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
print("Number of parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

# Stage 1
model.requires_grad_(False)
model.fc.requires_grad_(True)
optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=1e-4)

# Stage 2
backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
optimizer_fine = torch.optim.AdamW([{"params": backbone, "lr": LR_BASE},
                                    {"params": model.fc.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
fine_tune_steps = (EPOCHS - FREEZE_EPOCHS) * len(train_loader)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fine, T_max=fine_tune_steps)

def predict(loader, rows):
    model.eval()
    total_loss = 0.0
    all_probs = []
    with torch.no_grad():
        for indices in loader:
            images, labels = read_batch(rows, indices, test_transform)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return total_loss / len(rows), np.concatenate(all_probs)

history_rows = []
best_f1, best_epoch, wait = -1.0, 0, 0
for epoch in range(1, EPOCHS + 1):
    if epoch == FREEZE_EPOCHS + 1:
        print("Stage 2: fine-tune the whole ResNet18")
        model.requires_grad_(True)
    optimizer = optimizer_head if epoch <= FREEZE_EPOCHS else optimizer_fine

    print(f"Epoch {epoch}/{EPOCHS}")
    start = time.time()
    model.train()   
    total_loss, total_correct = 0.0, 0
    for indices in tqdm(train_loader, desc="Training"):
        images, labels = read_batch(train_rows, indices, train_transform)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if epoch > FREEZE_EPOCHS:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    val_loss, val_probs = predict(val_loader, val_rows)
    val_pred = val_probs.argmax(axis=1)
    val_true = val_rows["class_index"].to_numpy()
    row = {
        "epoch": epoch,
        "train_loss": total_loss / len(train_rows),
        "train_acc": total_correct / len(train_rows),
        "val_loss": val_loss,
        "val_acc": np.mean(val_pred == val_true),
        "val_f1": f1_score(val_true, val_pred, average="macro", zero_division=0)}
    history_rows.append(row)
    print(f"Loss: {row['train_loss']:.4f} | Accuracy: {row['train_acc']:.4f} | "
          f"Val Loss: {row['val_loss']:.4f} | Val Accuracy: {row['val_acc']:.4f} | "
          f"Val F1: {row['val_f1']:.4f} | Time: {time.time() - start:.0f}s")

    if row["val_f1"] > best_f1:
        best_f1, best_epoch, wait = row["val_f1"], epoch, 0
        best_weights = copy.deepcopy(model.state_dict())
    elif epoch > FREEZE_EPOCHS:   # early stopping only counts the fine-tuning epochs
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping: Macro-F1 did not improve.")
            break

model.load_state_dict(best_weights)
torch.save(model, OUT_DIR / "resnet18_model.pt")
history = pd.DataFrame(history_rows)
history.to_csv(OUT_DIR / "history.csv", index=False)

# Evaluate the best model on the validation and test sets
results = {"model": "resnet18", "img_size": 224,
           "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, split, loader, rows in [(axes[0], "valid", val_loader, val_rows), (axes[1], "test", test_loader, test_rows)]:
    loss, probabilities = predict(loader, rows)
    predictions = probabilities.argmax(axis=1)
    true_labels = rows["class_index"].to_numpy()

    print("\nResult", split.upper())
    print(classification_report(true_labels, predictions, target_names=class_names,
                                digits=4, zero_division=0))
    report = classification_report(true_labels, predictions, target_names=class_names,
                                   output_dict=True, zero_division=0)
    results[split] = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]}

    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=class_names,
                                            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(split)

    if split == "test":
        output = rows.copy()
        output["predicted"] = [class_names[i] for i in predictions]
        output["confidence"] = probabilities.max(axis=1)
        output["correct"] = predictions == true_labels
        output.to_csv(OUT_DIR / "predictions_test.csv", index=False)

fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=130)

# Learning curves and results file
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
history.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0], title="Loss")
history.plot(x="epoch", y=["train_acc", "val_acc"], ax=axes[1], title="Accuracy")
history.plot(x="epoch", y="val_f1", ax=axes[2], title="Macro-F1")
for ax in axes:
    ax.axvline(best_epoch, linestyle="--", color="gray")
    ax.axvline(FREEZE_EPOCHS + 0.5, linestyle=":", color="gray")
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curves.png", dpi=130)

with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Best epoch: {best_epoch} | Val Macro-F1: {best_f1:.4f}")
print("Results saved to:", OUT_DIR)