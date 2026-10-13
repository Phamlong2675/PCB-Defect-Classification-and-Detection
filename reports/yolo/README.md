# Hai checkpoint YOLOv8s dành cho test và vấn đáp

Repo kèm hai checkpoint tốt nhất được chọn **theo validation mAP@0.5:0.95**, cùng kết quả train và test. Không dùng test để chọn checkpoint.

| Giai đoạn | Run | Seed | Epoch tốt nhất | Validation mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Transfer (freeze=10) | 3 | 2052798190 | 23 | 75,059% |
| Fine-tuning (freeze=0) | 1 | 876847882 | 23 | 77,824% |

YOLOv8s có validation mAP@0.5:0.95 trung bình tốt nhất trong ba kiến trúc đã thử. Sau khi chọn kiến trúc, chọn run tốt nhất riêng cho từng giai đoạn. Fine-tuning Run 1 tiếp tục từ transfer Run 1; nó không tiếp tục từ checkpoint transfer Run 3 được xuất bản ở đây. Vì vậy, chênh lệch giữa hai hàng không phải phép so sánh cặp cùng seed.

## Chạy test ngay

Cài môi trường theo [README chính](../../README.md#installation), rồi chạy từ gốc repo:

```powershell
.\.venv\Scripts\python.exe .\PCB_learning\test_pcb.py
```

Mặc định test fine-tuning Run 1. Không cần train hoặc tải pretrained weights. Script tự chọn GPU CUDA hoặc CPU, tự cấu hình đường dẫn dataset.

Muốn test transfer, sửa ba biến trong `PCB_learning/test_pcb.py`:

```python
MODEL = "yolov8s"
PHASE = "transfer"
SEED = 2052798190
```

Muốn trở lại fine-tuning, dùng `PHASE="finetune"`, `SEED=876847882`. Mỗi lần test tạo output mới trong `PCB_learning/testing/`.

## Kết quả test đã kiểm tra trên dataset hiện tại

Đã chạy lại hai checkpoint trên `Data/processed/detection/test`: **150 ảnh, 984 bounding box**. Các chỉ số dưới đây là kết quả từng checkpoint, không phải trung bình ba run.

| Checkpoint | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
| --- | --- | --- | --- | --- |
| Transfer Run 3 | 94,53 | 94,84 | 97,26 | 76,98 |
| Fine-tuning Run 1 | 97,76 | 94,07 | 97,66 | 78,66 |

Kết quả đầy đủ nằm trong:

- [Transfer Run 3](test/yolov8s_transfer_seed2052798190): `metrics.csv`, `per_class.csv`, đường cong PR/F1/P/R, confusion matrix và ảnh nhãn/dự đoán.
- [Fine-tuning Run 1](test/yolov8s_finetune_seed876847882): cùng cấu trúc.

Chỉ số CSV ở thang 0–1; nhân 100 để đổi sang %. PR curve dùng IoU=0.5. Confusion matrix được tạo với confidence=0.001 của script test; không so trực tiếp số ô với ma trận dùng ngưỡng 0.25. Tham số NMS `iou=0.7` khác với các ngưỡng IoU dùng tính mAP.

## Các file để trình bày khi vấn đáp

- Checkpoint: `PCB_learning/runs/yolov8s_640_aug_frozen_run3/weights/best.pt` và `PCB_learning/runs/yolov8s_640_aug_finetune_run1/weights/best.pt`.
- `results.csv` và `results.png` trong mỗi run: loss train/validation và metric validation qua epoch; không phải metric test.
- `args.yaml`: cấu hình gốc của lần train; đường dẫn máy cũ trong file chỉ phục vụ ghi nhận, script test dùng dataset hiện tại.
- PR curve và confusion matrix nằm trong run thuộc validation; hình trong `reports/yolo/test/` thuộc test.
- [selected_checkpoints.csv](selected_checkpoints.csv): tiêu chí chọn, seed, epoch, đường dẫn và SHA-256 hai checkpoint.
- [evaluation_environment.json](evaluation_environment.json): môi trường test, tham số đánh giá và SHA-256 ảnh/nhãn test.

Các checkpoint được train trên bản dữ liệu cũ có 1.200 ảnh train. Dataset hiện tại loại một ảnh train lỗi, còn 1.199 ảnh; hai checkpoint không được train lại trong lần xuất bản này. Kết quả test ở trên được đánh giá trực tiếp trên dataset hiện tại. Khi train lại, kết quả có thể khác.

Chỉ hai `best.pt` và kết quả tương ứng được xuất bản; các checkpoint/run khác vẫn giữ trên máy gốc.
