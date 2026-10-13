from pathlib import Path
from ultralytics import YOLO
from pcb_augmentation import DEVICE, RandomRightAngleRotate, make_dataset_yaml, make_random_seed

MODE = "transfer"  # transfer / finetune
RUN_ID = 1  # 1 / 2 / 3; giữ cùng ID khi chuyển transfer -> finetune.
ROOT = Path(__file__).resolve().parent
BEST_PATH_FILE = ROOT / f"yolov8s_p2_aug_transfer_best_run{RUN_ID}.txt"


def main():
    assert MODE in ("transfer", "finetune")
    seed = make_random_seed(RUN_ID)
    transfer = MODE == "transfer"
    # Clone repo dùng checkpoint transfer có sẵn; train mới ưu tiên file trỏ vừa lưu.
    weights = "yolov8s.pt" if transfer else (
        Path(BEST_PATH_FILE.read_text(encoding="utf-8").strip()) if BEST_PATH_FILE.is_file()
        else ROOT / "runs" / f"yolov8s_p2_640_aug_frozen_run{RUN_ID}/weights/best.pt")
    if not transfer:
        assert weights.is_file(), f"Missing checkpoint: {weights}"
    data = make_dataset_yaml()
    model = (YOLO("yolov8s-p2.yaml").load(str(weights))
             if transfer else YOLO(str(weights)))
    assert model.model.model[0].conv.out_channels == 32, "Wrong model size"
    assert model.model.stride.tolist() == [4, 8, 16, 32], "Wrong detection heads"

    # Transfer đóng băng backbone; fine-tune mở backbone và giảm learning rate.
    epochs, lr0, freeze, stage = (30, 0.001, 10, "frozen") if transfer else (50, 0.0001, 0, "finetune")
    model.train(
        data=str(data), imgsz=640, batch=4, epochs=epochs, patience=10,
        freeze=freeze, optimizer="AdamW", lr0=lr0,
        warmup_epochs=3.0, warmup_bias_lr=0.0, resume=False,
        device=DEVICE, workers=0, amp=False, seed=seed, deterministic=True,
        augmentations=[RandomRightAngleRotate(p=1.0)], fliplr=0.5, flipud=0.5,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
        val=True, split="val", save=True, plots=True,
        project=str(ROOT / "runs"), name=f"yolov8s_p2_640_aug_{stage}_run{RUN_ID}", exist_ok=False,
    )
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    if transfer and best.is_file():
        BEST_PATH_FILE.write_text(str(best.resolve()), encoding="utf-8")
    print("Best weights:", best)


if __name__ == "__main__":
    main()
