# YOLOv8 PCB defect detection

Ba kiến trúc được dùng là YOLOv8n, YOLOv8s và YOLOv8s-P2. Mỗi thí nghiệm gồm transfer với backbone đóng băng, sau đó fine-tune từ checkpoint transfer tốt nhất. Tất cả lệnh dưới đây chạy tại **thư mục gốc repo** `Deep-Learning-PCB-Defect-Detection`.

## 1. Cài thư viện

Tạo môi trường Python 3.12, cài torch/torchvision theo GPU và chạy `python -m pip install -r requirements.txt` như hướng dẫn ở [README chính](../README.md#installation).

Các thư viện trực tiếp của phần YOLO:

| Thư viện | Vai trò |
| --- | --- |
| torch, torchvision | Tính toán, GPU và các phép xử lý detection |
| ultralytics 8.4.174 | Tạo model, train và test YOLOv8 |
| albumentations 2.0.8 | Xoay ảnh và bounding box theo góc vuông |
| PyYAML 6.0.3 | Đọc cấu hình dữ liệu |

Các module `csv`, `pathlib`, `os`, `secrets` thuộc thư viện chuẩn Python, không cần cài riêng. Tham khảo cách cài package tại [Ultralytics Quickstart](https://docs.ultralytics.com/quickstart/).

## 2. Chuẩn bị dữ liệu và pretrained weights

Dataset dùng trực tiếp từ `Data/processed/detection/` trong repo: **1.199 ảnh train, 150 ảnh validation và 150 ảnh test**, kích thước 640 x 640. Một ảnh train lỗi đã được loại bỏ khỏi bản 1.200 ảnh trước đây. Không cần khôi phục dataset cũ hoặc sửa đường dẫn thủ công.

Mỗi lần train/test, helper đọc `Data/processed/detection/data.yaml`, cập nhật `path` theo vị trí clone hiện tại và tạo `PCB_learning/pcb_640_local.yaml` (được Git bỏ qua). Nhãn là YOLO midpoint chuẩn hóa: `class_id x_center y_center width height`; thứ tự class giữ nguyên trong `data.yaml`.

Kết quả thí nghiệm đã ghi nhận trước đây dùng bản dữ liệu cũ. Ảnh bị loại bỏ có thể chỉ ảnh hưởng nhỏ, nhưng chưa đo lại mức ảnh hưởng; không đảm bảo tái tạo chính xác các chỉ số cũ khi train trên bản đã sửa.

Pretrained weights tự tải trong lần chạy transfer đầu tiên, nên cần Internet. Có thể tải trước từ thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
```

Hai file `yolov8n.pt`, `yolov8s.pt` nằm ở gốc repo và được Git bỏ qua. YOLOv8s-P2 tạo kiến trúc từ `yolov8s-p2.yaml` có sẵn trong Ultralytics, sau đó nạp các trọng số tương thích từ `yolov8s.pt`.

## 3. Chọn model, giai đoạn và run

| Model | File training | Tên dùng trong script test |
| --- | --- | --- |
| YOLOv8n | `yolo_pcb_aug.py` | `yolov8n` |
| YOLOv8s | `Yolo8s_pcb_aug.py` | `yolov8s` |
| YOLOv8s-P2 | `Yolo8p2_pcb_aug.py` | `yolov8s_p2` |

Trong file training cần chạy, chỉnh hai biến:

```python
MODE = "transfer"  # transfer / finetune
RUN_ID = 1         # 1 / 2 / 3
```

Cả ba file mặc định `MODE="transfer"`, `RUN_ID=1`, dùng được với clone mới. Code tự chọn GPU CUDA đầu tiên nếu có; nếu không sẽ dùng CPU (chậm hơn nhiều).

| Tham số | Transfer | Fine-tune |
| --- | --- | --- |
| Epoch tối đa | 30 | 50 |
| Learning rate ban đầu | 0.001 | 0.0001 |
| Freeze | 10 module đầu | 0, mở backbone |
| Patience | 10 | 10 |
| Batch / kích thước ảnh | 4 / 640 | 4 / 640 |
| Optimizer | AdamW | AdamW |

Augmentation gồm lật ngang và lật dọc, mỗi phép xác suất 0.5; xoay 90°, 180° hoặc 270° với xác suất bằng nhau. Thư viện cập nhật bounding box cùng ảnh. Các phép đổi màu, scale, translate, mosaic, mixup và các augmentation khác được tắt trong cấu hình training.

Seed được lưu một lần cho mỗi run, dùng chung giữa ba model và hai giai đoạn:

| RUN_ID | Seed đã dùng |
| --- | --- |
| 1 | 876847882 |
| 2 | 842398139 |
| 3 | 2052798190 |

Giữ ba file `pcb_aug_seed_run*.txt` để tái lập thí nghiệm. Nếu một file chưa tồn tại, helper sẽ sinh seed mới và lưu lại; điều này không thay đổi seed của checkpoint cũ.

## 4. Chạy transfer rồi fine-tune

Chạy lệnh tương ứng sau khi chỉnh `MODE` và `RUN_ID`:

```powershell
.\.venv\Scripts\python.exe .\PCB_learning\yolo_pcb_aug.py
.\.venv\Scripts\python.exe .\PCB_learning\Yolo8s_pcb_aug.py
.\.venv\Scripts\python.exe .\PCB_learning\Yolo8p2_pcb_aug.py
```

Ba dòng trên là ba lựa chọn model; chỉ chạy dòng cần dùng. Sau khi transfer hoàn tất, đổi `MODE="finetune"`, giữ nguyên `RUN_ID`, rồi chạy lại cùng file.

Transfer tự ghi đường dẫn `best.pt` vào `<model>_aug_transfer_best_run<ID>.txt`. Fine-tune ưu tiên file này; nếu chưa có, dùng `runs/<model>_640_aug_frozen_run<ID>/weights/best.pt`. Cả hai cách đều bắt đầu optimizer/lịch học mới với `resume=False`. Không đưa các file đường dẫn tuyệt đối này lên Git. Repo chỉ kèm transfer YOLOv8s Run 3; các model/run khác cần train transfer hoặc nhận checkpoint riêng trước khi fine-tune.

Kết quả được lưu dưới `PCB_learning/runs/`:

```text
<model>_640_aug_frozen_run<ID>/
<model>_640_aug_finetune_run<ID>/
  weights/best.pt
  weights/last.pt
  args.yaml
  results.csv
  results.png
```

`best.pt` được chọn theo validation mAP@0.5:0.95. Early stopping có thể kết thúc trước số epoch tối đa khi điểm tốt nhất không cải thiện trong 10 epoch. Khi chạy lại cùng tên, Ultralytics tạo folder có hậu tố; file trỏ transfer nhớ đường dẫn thực tế đó.

## 5. Test một checkpoint

Trong `test_pcb.py`, chỉnh:

```python
MODEL = "yolov8s"   # yolov8n / yolov8s / yolov8s_p2
PHASE = "finetune"  # transfer / finetune
SEED = 876847882
```

`SEED` dùng để chọn run đã train, không dùng để tạo lại model. Script đối chiếu seed trong checkpoint trước khi test. Mặc định trên là YOLOv8s fine-tune Run 1 được chọn theo validation trong thí nghiệm đã thực hiện.

```powershell
.\.venv\Scripts\python.exe .\PCB_learning\test_pcb.py
```

Test dùng `split="test"`, ảnh 640, batch 4 và tắt test-time augmentation. Kết quả lưu tại `PCB_learning/testing/<model>_<phase>_seed<seed>/`; chạy lại tạo folder mới.

| File | Nội dung |
| --- | --- |
| `metrics.csv` | Model, phase, run, seed, Precision, Recall, mAP50, mAP50–95 |
| `per_class.csv` | Số ảnh/box từng lớp, P/R/F1 và AP từng lớp |
| `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` | Các đường cong test khi `plots=True` |
| `confusion_matrix*.png` | Ma trận thường và chuẩn hóa khi `plots=True` |
| `val_batch*_labels.jpg`, `val_batch*_pred.jpg` | Nhãn thật và dự đoán mẫu khi `plots=True` |

Các chỉ số CSV nằm trong thang **0–1**, nhân 100 để trình bày %. Muốn chỉ lưu bảng, đổi `plots=True` thành `plots=False`.

PR curve của Ultralytics thể hiện IoU = 0.5. Script tối giản hiện dùng confidence 0.001 để tính mAP; confusion matrix cũng dùng ngưỡng này, không phải ngưỡng 0.25 của các hình report được tạo bằng validator riêng trước đây. Đồ thị loss và validation mAP theo epoch lấy từ `runs/.../results.png`, không phải từ test.

Script tìm checkpoint theo tên folder gốc ở trên. Nếu train lại tạo folder hậu tố như `-2`, cần trỏ `weights` trong script test tới đúng folder thực tế trước khi đánh giá.

## 6. Test ngay hai checkpoint tốt nhất đã có trong repo

Sau khi clone và cài thư viện, repo đã có hai checkpoint YOLOv8s được chọn theo validation mAP@0.5:0.95:

| Phase | Run | Seed | Validation mAP@0.5:0.95 | Epoch tốt nhất |
| --- | --- | --- | --- | --- |
| Transfer | 3 | 2052798190 | 75,059% | 23 |
| Fine-tuning | 1 | 876847882 | 77,824% | 23 |

Fine-tuning Run 1 được train tiếp từ transfer Run 1, không phải transfer Run 3. Đây là hai checkpoint tốt nhất được chọn độc lập theo từng giai đoạn.

```text
PCB_learning/runs/yolov8s_640_aug_frozen_run3/weights/best.pt
PCB_learning/runs/yolov8s_640_aug_finetune_run1/weights/best.pt
```

Chạy ngay fine-tuning bằng lệnh ở mục 5, không cần train hoặc tải pretrained weights. Muốn test transfer, đặt `MODEL="yolov8s"`, `PHASE="transfer"`, `SEED=2052798190`, rồi chạy cùng lệnh. Máy không có CUDA vẫn test được bằng CPU.

Kết quả train gốc gồm `results.csv`, `results.png`, `args.yaml`, đường cong PR, confusion matrix và ảnh mẫu trong hai thư mục run. Những đường cong/ma trận trong run thuộc validation. Kết quả test mới trên dataset hiện tại nằm ở [reports/yolo/test](../reports/yolo/test); xem [giải thích kết quả và cách chọn checkpoint](../reports/yolo/README.md).

Nếu nhận checkpoint khác riêng, giữ seed file tương ứng và đặt `best.pt` vào đúng cấu trúc run. Muốn fine-tune checkpoint transfer từ nơi khác, có thể cập nhật file `<model>_aug_transfer_best_run<ID>.txt` thành đường dẫn tuyệt đối tới checkpoint, lưu UTF-8 không BOM.

## 7. File nên đưa lên GitHub

Đưa lên: ba file training, `pcb_augmentation.py`, `test_pcb.py`, README và ba seed file; kèm README chính, requirements và .gitignore ở gốc repo.

Ngoại lệ đã đưa lên Git: hai `best.pt` YOLOv8s ở mục 6, kết quả train gốc tương ứng và kết quả test đã chọn trong `reports/yolo/test/`. Các checkpoint/run còn lại, `last.pt`, `.venv/`, output test mới, label cache, `pcb_640_local.yaml` và file trỏ checkpoint giữ local.
