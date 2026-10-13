import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

# Configuration
DATA_DIR = Path("Data/processed/classification")
MODEL_FILE = Path("runs/resnet18/resnet18_model.pt")   
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = manifest.drop_duplicates("class_index")
class_names = class_names.sort_values("class_index")["class_name"].tolist()
num_classes = len(class_names)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_image(path):
    with Image.open(path) as image:
        if image.size != (224, 224):
            raise ValueError(f"Invalid image size: {path}")
        image = image.convert("L").convert("RGB")
        return test_transform(image)

model = torch.load(MODEL_FILE, map_location=device, weights_only=False)
model.eval()
print("Loaded model:", MODEL_FILE, "| Device:", device)

def predict(paths):
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            images = torch.stack([read_image(p) for p in paths[i:i + BATCH_SIZE]]).to(device)
            all_probs.append(torch.softmax(model(images), dim=1).cpu().numpy())
    probabilities = np.concatenate(all_probs)
    return np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)

if len(sys.argv) > 1:
    predictions, confidence = predict(sys.argv[1:])
    for path, label, conf in zip(sys.argv[1:], predictions, confidence):
        print(f"{path}: {class_names[label]} ({conf:.1%})")
    sys.exit()

rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
paths = [DATA_DIR / path for path in rows["crop_path"]]
true_labels = rows["class_index"].to_numpy()

start = time.time()
predictions, confidence = predict(paths)
print(f"Predicted {len(paths)} test images in {time.time() - start:.0f}s")

examples = pd.DataFrame({
    "image": rows["crop_name"],
    "true": [class_names[i] for i in true_labels],
    "predicted": [class_names[i] for i in predictions],
    "confidence": np.round(confidence, 4)})
print(examples.head(10).to_string(index=False))

# Evaluate the predictions
report = classification_report(
    true_labels, predictions,
    labels=list(range(num_classes)), target_names=class_names,
    output_dict=True, zero_division=0)
print("\nTEST RESULT: ResNet18 (transfer learning + fine-tuning)")
print(classification_report(
    true_labels, predictions,
    labels=list(range(num_classes)), target_names=class_names,
    digits=4, zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(true_labels, predictions, labels=list(range(num_classes))),
                   index=class_names, columns=class_names))
print(f"\nAccuracy: {report['accuracy']:.4f} | Precision: {report['macro avg']['precision']:.4f} | "
      f"Recall: {report['macro avg']['recall']:.4f} | F1-score: {report['macro avg']['f1-score']:.4f} (macro)")
print("Wrong predictions:", int(np.sum(predictions != true_labels)), "of", len(true_labels))