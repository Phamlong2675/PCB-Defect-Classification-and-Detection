"""Test một checkpoint theo model, phase và seed; lưu bảng vào testing."""
import csv
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from pcb_augmentation import DEVICE, make_dataset_yaml

MODEL = "yolov8s"   # yolov8n / yolov8s / yolov8s_p2
PHASE = "finetune"  # transfer / finetune
SEED = 876847882    # Mặc định: YOLOv8s fine-tune run 1, chọn theo validation.
ROOT = Path(__file__).resolve().parent


def main():
    run = next((i for i in (1, 2, 3) if int((ROOT / f"pcb_aug_seed_run{i}.txt").read_text()) == SEED), None)
    assert run is not None, f"Seed {SEED} not found in pcb_aug_seed_run*.txt"
    stage = {"transfer": "frozen", "finetune": "finetune"}[PHASE]
    weights = ROOT / "runs" / f"{MODEL}_640_aug_{stage}_run{run}" / "weights/best.pt"
    assert weights.is_file(), f"Missing checkpoint: {weights}"
    model = YOLO(str(weights))
    assert model.ckpt["train_args"]["seed"] == SEED, "Checkpoint seed mismatch"
    # Tự tạo folder mới nếu đã test cùng model/phase/seed trước đó.
    output = increment_path(ROOT / "testing" / f"{MODEL}_{PHASE}_seed{SEED}", mkdir=True)
    m = model.val(
        data=str(make_dataset_yaml()), split="test", imgsz=640, batch=4, device=DEVICE, workers=0,
        rect=False, quantize=32, augment=False, conf=0.001, iou=0.7, max_det=300,
        plots=True, verbose=False, project=str(output.parent), name=output.name, exist_ok=True,
    )
    row = dict(model=MODEL, phase=PHASE, run=run, seed=SEED,
               precision=float(m.box.mp), recall=float(m.box.mr),
               mAP50=float(m.box.map50), mAP50_95=float(m.box.map))
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    (output / "per_class.csv").write_text(m.to_csv(), encoding="utf-8-sig")
    print("Results:", output)  # Chỉ số 0–1; nhân 100 để đổi sang %.


if __name__ == "__main__":
    main()
