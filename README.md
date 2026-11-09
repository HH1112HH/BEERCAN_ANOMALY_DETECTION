# BeerCan DRAEM - Automated Visual Inspection System

Hệ thống kiểm tra lỗi bề mặt đáy lon bia tự động ứng dụng Trí tuệ Nhân tạo (Computer Vision & Deep Learning) kết hợp băng tải và cơ cấu gạt phân loại lon lỗi sử dụng ESP8266 & Servo MG996R.

---

## 1. Kiến trúc Hệ thống

- **Phát hiện lon (Can Detection & Alignment):** YOLOv8 phát hiện tọa độ lon theo thời gian thực và thực hiện cắt tròn viền đáy lon chính xác (snug circular crop).
- **Phát hiện bất thường (Anomaly Detection):** Mô hình DRAEM (Discriminative Reconstructive Anomaly Embedding) tự giám sát, tái tạo ảnh chuẩn và sinh Heatmap lỗi bề mặt (vết móp, trầy xước, dị vật).
- **Phân loại & Ra quyết định (Classifier):** Ngưỡng Anomaly Score (mặc định 0.25) phân định lon GOOD và NG.
- **Điều khiển Băng tải & Phân loại (Hardware Control):** ESP8266 điều khiển Servo kim loại MG996R gạt lon NG với độ trễ băng tải 900ms.
- **Giao diện Giám sát (Dashboard UI):** FastAPI phục vụ giao diện Web thời gian thực tại `http://localhost:5004`, hỗ trợ xem trực tiếp camera, heatmap, lịch sử kiểm tra và xuất báo cáo CSV/SQLite.

---

## 2. Cấu trúc Dự án

```text
d:\bc_moi\
├── main.py                     # Khởi động hệ thống kiểm tra chính (FastAPI + WebSocket + Pipeline)
├── cap.py                      # Công cụ chụp ảnh thu thập mẫu từ camera
├── cap.bat / cap.ps1           # Phím tắt chạy nhanh công cụ chụp ảnh: .\cap.bat [ok|ng]
├── prepare_dataset.py          # Tiền xử lý và tăng cường dữ liệu huấn luyện DRAEM
├── config.yaml                 # File cấu hình toàn bộ hệ thống (Camera, Model, ESP, Server)
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── dataset_labeled.zip         # File nén dataset sẵn sàng upload Colab
├── HUONG_DAN_CHAY.md           # Hướng dẫn chạy nhanh hệ thống
├── README.md                   # Tài liệu tổng quan dự án
│
├── src/                        # Thư mục mã nguồn các module dịch vụ lõi
│   ├── __init__.py
│   ├── camera_service.py       # Quản lý luồng video RTSP từ camera công nghiệp / webcam
│   ├── detector.py             # Nhận diện vị trí lon và cắt tròn ôm sát bằng YOLOv8
│   ├── draem_model.py          # Mô hình DRAEM (Reconstructor & Discriminative Sub-network)
│   ├── classifier.py           # Logic phân loại GOOD / NG dựa trên điểm số bất thường
│   ├── conveyor_controller.py  # Module giao tiếp HTTP/Serial với ESP8266
│   ├── db_manager.py           # Quản lý cơ sở dữ liệu SQLite lưu trữ lịch sử kiểm tra
│   └── image_logger.py         # Lưu trữ hình ảnh chụp lon, ROI và Heatmap
│
├── static/                     # Giao diện Web Dashboard (HTML5, Modern CSS, JS)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── models/                     # Trọng số mô hình AI
│   ├── draem.pth               # Trọng số mô hình DRAEM chính thức
│   └── weights.pt              # Trọng số mô hình YOLOv8 phát hiện lon
│
├── esp_firmware/               # Mã nguồn nạp vi điều khiển ESP8266
│   ├── test1/test1.ino         # Firmware chính thức (MG996R, delay gạt 900ms)
│   ├── test1_backup2/          # Bản sao lưu firmware 2
│   └── test1_backup3/          # Bản sao lưu firmware 3
│
├── dataset/                    # Dữ liệu ảnh gốc thu thập từ camera
│   ├── train_ok/               # Mẫu lon đạt chuẩn
│   └── test_ng/                # Mẫu lon lỗi thực tế
│
├── dataset_labeled/            # Dữ liệu đã qua cắt viền và tăng cường sẵn sàng train
│
├── notebooks/                  # Jupyter Notebooks huấn luyện mô hình
│   ├── Draem_T11.ipynb         # Notebook huấn luyện DRAEM trên Google Colab / GPU
│   └── Draem_T11_2500.ipynb    # Checkpoint thực nghiệm huấn luyện 2500 epochs
│
├── captures/                   # Ảnh chụp kiểm tra thực tế trong quá trình vận hành
├── results/                    # Cơ sở dữ liệu SQLite (inspection_database.db) và báo cáo CSV
├── report_images/              # Biểu đồ và hình ảnh trực quan phục vụ báo cáo khoa học
│   └── calibration/            # Ảnh hiệu chuẩn bán kính cắt và độ sáng
└── docs/                       # Tài liệu kỹ thuật chi tiết
```

---

## 3. Hướng dẫn Vận hành

### Bước 1: Kích hoạt môi trường và cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình thông số
Kiểm tra file `config.yaml` để đảm bảo:
- `camera.rtsp_url`: Đúng đường dẫn RTSP của camera.
- `esp.ip`: Địa chỉ IP của ESP8266 trên mạng nội bộ.
- `draem.threshold`: Ngưỡng quyết định lon NG (khuyến nghị 0.25).

### Bước 3: Khởi động hệ thống kiểm tra
```bash
python main.py
```

### Bước 4: Mở Dashboard
Truy cập trình duyệt: `http://localhost:5004`
- Theo dõi luồng camera trực tiếp và trạng thái lon (GOOD / NG).
- Xem ảnh Heatmap phân tích vùng lỗi.
- Theo dõi thống kê tổng sản lượng, tỷ lệ đạt/lỗi và xuất báo cáo.

---

## 4. Công cụ Thu thập & Xử lý Dữ liệu

- **Chụp ảnh thu thập mẫu:**
  ```bash
  python cap.py        # Chụp lon OK (hoặc gõ .\cap.bat)
  python cap.py ng     # Chụp lon NG (hoặc gõ .\cap.bat ng)
  ```
- **Tiền xử lý & Tăng cường dữ liệu:**
  ```bash
  python prepare_dataset.py
  ```
