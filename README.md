# 🍺 BeerCan DRAEM - Automated Visual Inspection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![ESP8266](https://img.shields.io/badge/Hardware-ESP8266%20%7C%20MG996R-E7352C)](https://www.espressif.com/)

Hệ thống kiểm tra lỗi bề mặt đáy lon bia tự động ứng dụng Trí tuệ Nhân tạo (**Computer Vision & Deep Learning - DRAEM**) kết hợp băng chuyền công nghiệp và cơ cấu gạt phân loại lon lỗi sử dụng vi điều khiển **ESP8266** và động cơ Servo kim loại **MG996R**.

---

## 📌 Tính Năng Nổi Bật

- **Định vị & Cắt tròn ôm sát 100% (Snug Circular Crop):** Sử dụng YOLOv8 phát hiện tọa độ lon theo thời gian thực và thuật toán Hough Circle Gradient che phủ viền đen, triệt tiêu hoàn toàn nhiễu từ thành băng chuyền.
- **Phát hiện bất thường với DRAEM:** Mô hình tự giám sát (Reconstructive + Discriminative Sub-networks) phát hiện chính xác các vết trầy xước, móp méo, dị vật, vết bẩn ở đáy lon và sinh ảnh **Heatmap** trực quan hóa vùng lỗi.
- **Phân loại & Ra quyết định tự động:** So sánh điểm số bất thường (Anomaly Score) với ngưỡng chuẩn (`threshold = 0.25`) để phân định tức thời `GOOD` hoặc `NG`.
- **Cơ cấu gạt phân loại MG996R tối ưu:** ESP8266 điều khiển Servo MG996R (nhông kim loại) với thuật toán di chuyển 2 giai đoạn (Snappy Stroke) và độ trễ hành trình lon được hiệu chuẩn chính xác `900ms`.
- **Web Dashboard thời gian thực:** Giao diện điều khiển trực quan tại cổng `5004`, truyền luồng video camera trực tiếp, hiển thị kết quả kiểm tra, ảnh Heatmap, đếm sản lượng và quản lý phiên kiểm tra.
- **Lưu trữ & Báo cáo hoàn chỉnh:** Ghi lại toàn bộ lịch sử kiểm tra vào cơ sở dữ liệu SQLite (`inspection_database.db`) và hỗ trợ xuất báo cáo định dạng CSV theo từng ca/phiên sản xuất.

---

## 🏗️ Kiến Trúc Hệ Thống

```
 [ Camera IP Dahua RTSP ]
            │
            ▼ (Luồng video 1280x720)
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND (Python)                       │
│                                                             │
│  [1. CameraService]  ──> Đọc luồng RTSP thời gian thực      │
│  [2. CanDetector]    ──> YOLOv8 phát hiện đáy lon & cắt tròn│
│  [3. DRAEM Model]    ──> Tái tạo ảnh & sinh Heatmap lỗi     │
│  [4. Classifier]     ──> Đánh giá điểm Score vs Ngưỡng 0.25 │
│  [5. ImageLogger]    ──> Lưu ảnh chụp raw, ROI & Heatmap    │
│  [6. DatabaseManager]──> Lưu kết quả vào SQLite Database    │
└─────────────────────────────────────────────────────────────┘
       │                                          │
       ▼ (Quyết định GOOD / NG)                   ▼ (WebSocket & REST API)
┌──────────────────────────────┐          ┌───────────────────────────┐
│     PHẦN CỨNG BĂNG TẢI       │          │   GIAO DIỆN WEB DASHBOARD │
│  ESP8266 NodeMCU             │          │   (http://localhost:5004) │
│  + Servo MG996R (Chân D5)    │          │  - Video Stream trực tiếp │
│  - Lon GOOD: Đi thẳng        │          │  - Chi tiết Anomaly Score │
│  - Lon NG: Gạt sang làn lỗi  │          │  - Biểu đồ & Lịch sử kiểm │
└──────────────────────────────┘          └───────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục

Dự án được tổ chức theo chuẩn kiến trúc module phân lớp sạch sẽ:

```text
d:\bc_moi\
├── main.py                     # Entry point chính: Khởi động FastAPI Server & Pipeline
├── cap.py                      # Công cụ chụp ảnh thu thập mẫu dataset từ camera
├── cap.bat / cap.ps1           # Phím tắt chạy nhanh công cụ chụp: .\cap.bat [ok|ng]
├── prepare_dataset.py          # Tiền xử lý, cắt viền tròn & tăng cường dữ liệu DRAEM
├── config.yaml                 # Cấu hình hệ thống (Camera RTSP, AI Models, ESP, Server)
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── dataset_labeled.zip         # Tập dữ liệu mẫu đã đóng gói sẵn sàng nạp Colab
├── HUONG_DAN_CHAY.md           # Cẩm nang hướng dẫn vận hành chi tiết từ A - Z
├── README.md                   # Tài liệu tổng quan dự án
│
├── src/                        # Các module dịch vụ lõi của hệ thống
│   ├── __init__.py             # Định danh Python Package
│   ├── camera_service.py       # Quản lý luồng RTSP, buffer khung hình camera
│   ├── detector.py             # YOLOv8 nhận diện đáy lon & thuật toán cắt tròn ôm sát
│   ├── draem_model.py          # Kiến trúc mô hình DRAEM (Reconstructor & Discriminator)
│   ├── classifier.py           # Logic phân loại GOOD / NG, quản lý bộ đếm lon
│   ├── conveyor_controller.py  # Điều khiển giao tiếp socket/HTTP với ESP8266
│   ├── db_manager.py           # Quản lý cơ sở dữ liệu SQLite và truy vấn thống kê
│   └── image_logger.py         # Lưu trữ ảnh chụp kiểm tra, ROI và Heatmap màu JET
│
├── static/                     # Giao diện Web Dashboard người dùng
│   ├── index.html              # Layout bảng điều khiển và bảng thống kê
│   ├── app.js                  # Xử lý WebSocket, cập nhật trạng thái và âm thanh cảnh báo
│   └── style.css               # Phong cách giao diện Dark Mode công nghiệp hiện đại
│
├── models/                     # Trọng số mô hình AI đã huấn luyện
│   ├── draem.pth               # Checkpoint mô hình DRAEM chính thức (GPU CUDA)
│   └── weights.pt              # Trọng số YOLOv8 segment nhận diện đáy lon
│
├── esp_firmware/               # Mã nguồn firmware nạp cho vi điều khiển ESP8266
│   ├── test1/test1.ino         # Firmware chính thức (MG996R, delay gạt 900ms)
│   ├── test1_backup2/          # Bản firmware dự phòng 2
│   └── test1_backup3/          # Bản firmware dự phòng 3
│
├── dataset/                    # Dữ liệu ảnh gốc thu thập từ camera
│   ├── train_ok/               # Ảnh lon đạt chuẩn phục vụ huấn luyện
│   └── test_ng/                # Ảnh lon lỗi thực tế phục vụ kiểm thử
│
├── dataset_labeled/            # Dữ liệu đã tiền xử lý cắt tròn sẵn sàng huấn luyện
├── notebooks/                  # Jupyter Notebooks huấn luyện trên GPU / Google Colab
│   ├── Draem_T11.ipynb         # Notebook huấn luyện DRAEM hoàn chỉnh
│   └── Draem_T11_2500.ipynb    # Checkpoint thực nghiệm 2500 epochs
│
├── captures/                   # Thư mục lưu trữ ảnh chụp kiểm tra thực tế
├── results/                    # Cơ sở dữ liệu SQLite và các file báo cáo CSV
├── report_images/              # Đồ thị, ma trận nhầm lẫn và ảnh phục vụ báo cáo khoa học
│   └── calibration/            # 22 ảnh thực nghiệm hiệu chuẩn bán kính cắt tròn
└── docs/                       # Tài liệu thiết kế kỹ thuật và phần cứng
    ├── KIEN_TRUC_HE_THONG.md   # Chi tiết kiến trúc AI & Pipeline xử lý
    └── PHAN_CUNG_ESP8266.md    # Sơ đồ mạch, chân nối và hiệu chuẩn Servo MG996R
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Nhanh

### 1. Yêu Cầu Môi Trường
- **Hệ điều hành:** Windows 10/11 hoặc Linux.
- **Python:** 3.10 hoặc 3.11.
- **GPU (Khuyến nghị):** NVIDIA GPU với CUDA 11.8+ để tối ưu tốc độ xử lý thời gian thực.

### 2. Cài Đặt Thư Viện
Mở Terminal tại thư mục `d:\bc_moi`:
```bash
pip install -r requirements.txt
```

### 3. Cấu Hình Hệ Thống (`config.yaml`)
Chỉnh sửa file `config.yaml` phù hợp với mạng nội bộ:
```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0"
  inspect_delay: 0.18

esp:
  ip: "192.168.1.54"
  conveyor_port: 12345

draem:
  device: "cuda:0"
  model: "models/draem.pth"
  threshold: 0.25

yolo:
  model: "models/weights.pt"
  confidence: 0.65
  device: "cuda:0"

server:
  host: "0.0.0.0"
  port: 5004
```

### 4. Khởi Động Hệ Thống
```bash
python main.py
```

### 5. Mở Dashboard Giám Sát
Mở trình duyệt Web tại: **`http://localhost:5004`**

---

## 🛠️ Công Cụ Thu Thập & Huấn Luyện Dữ Liệu

1. **Chụp ảnh thu thập mẫu từ Camera:**
   ```bash
   python cap.py        # Chụp lon đạt chuẩn lưu vào dataset/train_ok
   python cap.py ng     # Chụp lon lỗi lưu vào dataset/test_ng
   # Hoặc sử dụng phím tắt:
   .\cap.bat
   .\cap.bat ng
   ```

2. **Tiền xử lý và tăng cường dữ liệu:**
   ```bash
   python prepare_dataset.py
   ```
   *Script tự động cắt viền tròn ôm khít, xoay góc, lật ảnh và đóng gói thành `dataset_labeled.zip`.*

3. **Huấn luyện mô hình:**
   - Mở file [`notebooks/Draem_T11.ipynb`](notebooks/Draem_T11.ipynb) trên Google Colab (chọn môi trường GPU T4 / A100).
   - Tải file `dataset_labeled.zip` lên Colab và bấm chạy toàn bộ notebook.
   - Tải file checkpoint trọng số sau huấn luyện (`draem.pth`) về đặt vào thư mục `models/draem.pth`.

---

## 📖 Tài Liệu Tham Khảo Thêm

- [📘 Hướng dẫn vận hành chi tiết (HUONG_DAN_CHAY.md)](HUONG_DAN_CHAY.md)
- [🔬 Kiến trúc AI & Pipeline xử lý ảnh chi tiết (docs/KIEN_TRUC_HE_THONG.md)](docs/KIEN_TRUC_HE_THONG.md)
- [🔌 Sơ đồ đấu nối phần cứng & Hiệu chuẩn ESP8266 / MG996R (docs/PHAN_CUNG_ESP8266.md)](docs/PHAN_CUNG_ESP8266.md)
