"""Dùng phép xoay có sẵn của Albumentations; chỉ chọn 90°, 180°, 270°."""

import os
import secrets
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
import torch
import yaml
from ultralytics.utils.torch_utils import init_seeds


DEVICE = 0 if torch.cuda.is_available() else "cpu"


def make_dataset_yaml():
    # Resolve the dataset from this clone, not the original author's computer.
    root = Path(__file__).resolve().parent
    dataset = root.parent / "Data/processed/detection"
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8-sig"))
    config["path"] = dataset.as_posix()
    local = root / "pcb_640_local.yaml"
    local.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return local


def make_random_seed(run_id):
    # Sinh ngẫu nhiên một lần cho mỗi lượt; cả 3 model và 2 giai đoạn dùng lại seed đó.
    assert run_id in (1, 2, 3), "RUN_ID must be 1, 2 or 3"
    seed_file = Path(__file__).resolve().parent / f"pcb_aug_seed_run{run_id}.txt"
    if not seed_file.is_file():
        seed_file.write_text(str(secrets.randbelow(2**31)), encoding="utf-8")
    seed = int(seed_file.read_text(encoding="utf-8").strip())
    init_seeds(seed, deterministic=True)  # Áp dụng trước cả lúc tạo model P2.
    print(f"Run {run_id} | seed: {seed}", flush=True)  # Cũng lưu trong runs/.../args.yaml.
    return seed


class RandomRightAngleRotate(A.RandomRotate90):
    """Kế thừa xử lý ảnh/box có sẵn; chỉ bỏ góc 0° khỏi lựa chọn."""

    def get_params(self):
        return {"factor": self.py_random.randint(1, 3)}  # 1/2/3 × 90°, mỗi góc 1/3.
