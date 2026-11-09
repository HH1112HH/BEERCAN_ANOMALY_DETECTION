# 📖 CẨM NANG HƯỚNG DẪN VẬN HÀNH HỆ THỐNG KIỂM TRA ĐÁY LON BIA

Tài liệu này cung cấp quy trình vận hành chi tiết từ A - Z cho toàn bộ hệ thống kiểm tra lỗi đáy lon bia tự động bằng mô hình AI DRAEM kết hợp băng chuyền và bộ gạt Servo ESP8266.

---

## MỤC LỤC
1. [Chuẩn bị phần cứng](#1-chuẩn-bị-phần-cứng)
2. [Cài đặt môi trường & cấu hình](#2-cài-đặt-môi-trường--cấu-hình)
3. [Quy trình khởi động hệ thống](#3-quy-trình-khởi-động-hệ-thống)
4. [Sử dụng Web Dashboard](#4-sử-dụng-web-dashboard)
5. [Thu thập & Tiền xử lý dữ liệu mới](#5-thu-thập--tiền-xử-lý-dữ-liệu-mới)
6. [Nạp lại Firmware cho ESP8266](#6-nạp-lại-firmware-cho-esp8266)
7. [Bảng khắc phục sự cố (Troubleshooting)](#7-bảng-khắc-phục-sự-cố-troubleshooting)

---

## 1. CHUẨN BỊ PHẦN CỨNG

Trước khi bật nguồn máy tính và chạy phần mềm, hãy kiểm tra các kết nối vật lý:

1. **Camera IP Dahua:**
   - Cắm dây mạng LAN nối camera với router/switch chung dải mạng với máy tính.
   - Đèn hồng ngoại / đèn LED chiếu sáng đáy lon đã bật ổn định.
   - Vị trí camera chiếu thẳng góc 90 độ xuống đáy lon trên băng tải.
2. **Bộ điều khiển ESP8266 NodeMCU & Servo MG996R:**
   - ESP8266 cắm cáp Micro-USB cấp nguồn (hoặc nguồn 5V ngoài).
   - **Động cơ Servo MG996R:** Đảm bảo được cấp nguồn 5V riêng (dòng tối thiểu 2A), **chung chân GND** với ESP8266 để tránh sụt áp gây reset ESP.
   - Chân tín hiệu Servo cắm vào chân **D5** (GPIO14) của ESP8266.
   - Tay gạt nằm ở vị trí chờ (Home 0 độ), không cản trở hành lang di chuyển của lon bình thường.
3. **Băng chuyền:**
   - Đã cắm nguồn motor băng chuyền, tốc độ quay ổn định, không bị giật.

---

## 2. CÀI ĐẶT MÔI TRƯỜNG & CẤU HÌNH

### Bước 2.1: Mở Terminal tại thư mục dự án
Đường dẫn dự án: `d:\bc_moi`

### Bước 2.2: Cài đặt thư viện phụ thuộc (nếu chưa cài)
```bash
pip install -r requirements.txt
```

### Bước 2.3: Kiểm tra file `config.yaml`
Mở file `config.yaml` và kiểm tra 3 thông số quan trọng:
```yaml
camera:
  # Địa chỉ RTSP camera thực tế (IP, user, password)
  rtsp_url: "rtsp://admin:L27B3642@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0"
  inspect_delay: 0.18    # Độ trễ chụp sau khi lon vào vùng nhận diện

esp:
  ip: "192.168.1.54"     # Địa chỉ IP của ESP8266 nhận từ router WiFi
  conveyor_port: 12345   # Cổng socket giao tiếp

draem:
  device: "cuda:0"       # Sử dụng GPU CUDA (hoặc 'cpu' nếu máy không có GPU)
  model: "models/draem.pth"
  threshold: 0.25        # Ngưỡng phân định lon NG (lỗi nếu Score >= 0.25)
```

---

## 3. QUY TRÌNH KHỞI ĐỘNG HỆ THỐNG

### Bước 3.1: Chạy Server chính
Tại Terminal thư mục `d:\bc_moi`, gõ lệnh:
```bash
python main.py
```

### Bước 3.2: Quan sát log khởi động
Màn hình console sẽ xuất hiện các thông báo thành công:
```text
[camera_service] INFO: Connected to camera: rtsp://...
[detector] INFO: YOLO segment model loaded on cuda:0
[draem_model] INFO: Loading DRAEM checkpoint from models/draem.pth onto cuda:0...
[draem_model] INFO: DRAEM loaded successfully! Model threshold: 0.2246, Active: 0.2500
[main] INFO: ESP connected at 192.168.1.54:12345
[db_manager] INFO: Database initialized at results\inspection_database.db
INFO: Uvicorn running on http://0.0.0.0:5004 (Press CTRL+C to quit)
```

---

## 4. SỬ DỤNG WEB DASHBOARD

Mở trình duyệt Web (Chrome, Edge) và truy cập: **`http://localhost:5004`**

### Các khu vực chính trên giao diện:
1. **Live Camera Feed (Khung hình trực tiếp):**
   - Hiển thị luồng video thời gian thực từ camera kèm khung nhận diện màu xanh/đỏ quanh đáy lon.
2. **Inspection Result (Kết quả lon mới nhất):**
   - **Lon GOOD (Đạt):** Khung viền xanh lá, hiển thị điểm bất thường thấp (ví dụ: `Score: 0.142`).
   - **Lon NG (Lỗi):** Khung viền đỏ nhấp nháy, phát âm thanh cảnh báo "Beep", hiển thị điểm bất thường cao (ví dụ: `Score: 0.785`).
3. **Heatmap Display (Bản đồ nhiệt vùng lỗi):**
   - Trực quan hóa màu (bảng màu JET: Đỏ/Vàng là vị trí có dị vật, trầy xước, móp méo; Xanh dương là bề mặt nhôm sạch).
4. **Bảng thống kê ca sản xuất (Session Stats):**
   - **Total Cans:** Tổng số lon đã chạy qua hệ thống.
   - **Passed (GOOD):** Số lon đạt chuẩn & Tỷ lệ %.
   - **Defective (NG):** Số lon bị phân loại lỗi & Tỷ lệ %.
5. **Các nút điều khiển:**
   - **Start / Stop Auto:** Bật/tắt chế độ tự động phân loại.
   - **Reset Counter:** Tạo phiên làm việc mới (ca làm việc mới) và xóa bộ đếm về 0.
   - **Export CSV:** Tải báo cáo chi tiết toàn bộ các lon đã kiểm tra trong ca về máy.

---

## 5. THU THẬP & TIỀN XỬ LÝ DỮ LIỆU MỚI

Khi muốn bổ sung mẫu lon mới để huấn luyện nâng cao độ chính xác:

### Bước 5.1: Chụp ảnh lon đạt chuẩn (Train OK)
Đặt từng lon tốt lên băng tải, chạy lệnh:
```bash
python cap.py
# hoặc gõ: .\cap.bat
```
- Nhấn phím **SPACE** (Cách) để chụp từng ảnh. Ảnh sẽ tự động lưu vào `dataset/train_ok/`.
- Nhấn **ESC** hoặc phím **Q** để thoát.

### Bước 5.2: Chụp ảnh lon lỗi (Test NG)
Đặt các lon có lỗi (móp, xước, dính bẩn) lên băng tải, chạy lệnh:
```bash
python cap.py ng
# hoặc gõ: .\cap.bat ng
```
- Nhấn phím **SPACE** để chụp từng ảnh lỗi. Ảnh lưu vào `dataset/test_ng/`.

### Bước 5.3: Tiền xử lý, cắt viền tròn & đóng gói
Chạy lệnh tự động:
```bash
python prepare_dataset.py
```
- Script sẽ tự động nhận diện vòng tròn lon bia bằng giải thuật Hough Circle Gradient.
- Che phủ nền đen hoàn toàn viền ngoài, loại bỏ nhiễu băng chuyền.
- Tự động xoay góc và tăng cường dữ liệu lên hàng nghìn mẫu.
- Xuất file đóng gói: `dataset_labeled.zip`.

### Bước 5.4: Huấn luyện lại trên Google Colab
1. Mở file [`notebooks/Draem_T11.ipynb`](notebooks/Draem_T11.ipynb) trên Google Colab.
2. Chọn Runtime: **GPU (T4 hoặc A100)**.
3. Kéo thả file `dataset_labeled.zip` lên mục Files của Colab.
4. Chạy toàn bộ các ô (Run All).
5. Khi huấn luyện xong (sau ~100 epochs), tải file `draem.pth` về máy và ghi đè vào `d:\bc_moi\models\draem.pth`.

---

## 6. NẠP LẠI FIRMWARE CHO ESP8266

Nếu thay đổi vi điều khiển ESP8266 hoặc cần nạp lại code:
1. Mở phần mềm **Arduino IDE**.
2. Mở file: [`esp_firmware/test1/test1.ino`](esp_firmware/test1/test1.ino).
3. Chọn Board: **NodeMCU 1.0 (ESP-12E Module)**.
4. Chọn đúng cổng COM kết nối với ESP8266.
5. Kiểm tra thông tin WiFi ở đầu file:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
6. Các thông số vận hành động cơ MG996R đã được tối ưu hóa:
   ```cpp
   #define SERVO_PIN            D5     // Chân tín hiệu điều khiển Servo
   #define NG_EJECT_DELAY_MS    900    // Độ trễ từ lúc chụp đến điểm gạt lon
   #define SERVO_STRIKE_ANGLE   70     // Góc gạt lon lỗi
   #define SERVO_STRIKE_HOLD    450    // Thời gian giữ tay gạt (ms)
   #define SERVO_STRIKE_RETURN  400    // Thời gian thu tay gạt về (ms)
   ```
7. Bấm **Upload** để nạp chương trình vào ESP8266.

---

## 7. BẢNG KHẮC PHỤC SỰ CỐ (TROUBLESHOOTING)

| Tình trạng | Nguyên nhân có thể | Cách khắc phục |
| :--- | :--- | :--- |
| **Báo lỗi `Cannot connect to RTSP`** | - Camera chưa cắm nguồn hoặc lỏng cáp LAN.<br>- IP camera trong `config.yaml` bị sai. | - Dùng lệnh `ping 192.168.1.20` kiểm tra camera.<br>- Đảm bảo máy tính và camera cùng dải mạng. |
| **Báo lỗi `ESP not connected`** | - ESP8266 chưa kết nối WiFi.<br>- IP của ESP8266 bị router cấp mới. | - Mở Serial Monitor (baud 115200) của Arduino IDE để xem IP mới của ESP8266.<br>- Cập nhật IP mới vào mục `esp.ip` trong `config.yaml`. |
| **Servo không gạt lon lỗi** | - Lỏng dây tín hiệu D5.<br>- Nguồn 5V cho Servo bị yếu dòng. | - Cấp nguồn 5V/2A riêng cho Servo MG996R.<br>- Nhớ nối chung dây GND của nguồn Servo với chân GND của ESP8266. |
| **Servo gạt bị sớm hoặc trễ hơn lon** | Tốc độ băng chuyền thay đổi khiến thời gian lon di chuyển từ camera đến tay gạt lệch 900ms. | Chỉnh lại thông số `#define NG_EJECT_DELAY_MS` trong file `esp_firmware/test1/test1.ino` (tăng lên nếu gạt sớm, giảm đi nếu gạt trễ). |
| **Báo lỗi `Port 5004 already in use`** | Lần chạy trước chưa tắt hẳn, tiến trình cũ vẫn chiếm cổng 5004. | Mở PowerShell gõ lệnh:<br>`Stop-Process -Name python -Force`<br>Sau đó chạy lại `python main.py`. |
| **Màn hình Web không load video** | Trình duyệt bị cache hoặc WebSocket chưa kết nối. | Nhấn `Ctrl + F5` để tải lại trang Web Dashboard. |

---

> *Nếu cần dừng toàn bộ hệ thống đang chạy trong Terminal: Nhấn tổ hợp phím **Ctrl + C**.*
