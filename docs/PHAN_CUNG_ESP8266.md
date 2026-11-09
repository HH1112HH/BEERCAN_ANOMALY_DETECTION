# 🔌 TÀI LIỆU KỸ THUẬT: PHẦN CỨNG ESP8266 & CƠ CẤU GẠT SERVO MG996R

Tài liệu hướng dẫn sơ đồ chân, nguyên lý cấp nguồn và thuật toán điều khiển động cơ Servo kim loại **MG996R** kết nối với vi điều khiển **ESP8266 NodeMCU** phục vụ phân loại lon lỗi trên băng tải.

---

## 1. SO SÁNH PHẦN CỨNG: SERVO MG996R VS SG90

Trong các phiên bản thử nghiệm ban đầu, động cơ **SG90** (nhông nhựa) thường xuyên gặp các vấn đề:
- Lực kéo yếu ($1.8 \text{ kg}\cdot\text{cm}$), dễ bị kẹt khi gạt lon bia nặng.
- Dễ gãy bánh răng nhựa sau nhiều chu kỳ đập liên tục.
- Hành trình đập bị rung lắc, dễ trượt bước.

**Nâng cấp lên MG996R (Metal Gear - Nhông Kim Loại):**
- Lực kéo cực đại: $11.0 \text{ kg}\cdot\text{cm}$ tại 6V (gấp 6 lần SG90).
- Hệ thống bánh răng hoàn toàn bằng hợp kim thép bền bỉ, chống mài mòn.
- Ổ bi đôi (Dual Ball Bearings) giúp trục quay ổn định, không bị đảo khi đập nhanh.

---

## 2. SƠ ĐỒ ĐẤU NỐI DÂY (WIRING DIAGRAM)

> [!CAUTION]
> **QUAN TRỌNG VỀ NGUỒN CẤP:** Động cơ MG996R khi tải nặng có dòng điện đỉnh (Stall Current) lên tới **1.5A - 2.5A**. Tuyệt đối **KHÔNG** lấy nguồn 5V/VIN trực tiếp từ chân của ESP8266 để nuôi MG996R vì sẽ gây sụt áp làm ESP8266 tự động Reset liên tục! Phải dùng **bộ nguồn 5V riêng** và **nối chung Mass (GND)**.

```
       [ BỘ NGUỒN NGOÀI 5V / 2A-3A ]
             (+) 5V  ───> Dây ĐỎ (VCC) của Servo MG996R
             (-) GND ──┬> Dây NÂU (GND) của Servo MG996R
                       │
                       └─> Chân GND của ESP8266 NodeMCU (Nối chung GND)

       [ ESP8266 NodeMCU v3 ]
             Chân D5 (GPIO14) ───> Dây CAM (Signal PWM) của Servo MG996R
             Cổng Micro-USB   ───> Cắm sạc 5V cấp nguồn nuôi riêng cho ESP
```

---

## 3. THUẬT TOÁN ĐIỀU KHIỂN & HIỆU CHUẨN THỜI GIAN

Hành trình gạt lon NG được thiết kế theo thuật toán **Snappy Stroke (Đập dứt khoát 2 giai đoạn)** để đảm bảo gạt văng lon lỗi ra khỏi băng tải mà không cản trở lon đi ngay sau:

```
T0: Nhận lệnh "NG" từ Backend Python
 │
 ├──> [ Chờ trễ: NG_EJECT_DELAY_MS = 900 ms ]
 │    (Lon di chuyển từ vị trí Camera đến trước mặt tay gạt)
 │
T1: ĐẬP GẠT LON (Strike Phase)
 │   - Xoay nhanh từ 0 độ (Home) sang 70 độ (Strike)
 │   - Giữ nguyên vị trí trong SERVO_STRIKE_HOLD = 450 ms để hất văng lon sang làn NG
 │
T2: THU HỒI TAY GẠT (Return Phase)
 │   - Xoay thu hồi từ 70 độ về 0 độ (Home) trong SERVO_STRIKE_RETURN = 400 ms
 │
T3: SẴN SÀNG CHỜ LƯỢT TIẾP THEO
```

### Các thông số cốt lõi trong Firmware (`esp_firmware/test1/test1.ino`):

```cpp
// Cấu hình chân & tham số thời gian
#define SERVO_PIN            D5     // Chân tín hiệu điều khiển Servo
#define NG_EJECT_DELAY_MS    900    // Độ trễ băng tải từ điểm chụp đến điểm gạt (ms)
#define SERVO_HOME_ANGLE     0      // Vị trí nghỉ (thu sát thành băng tải)
#define SERVO_STRIKE_ANGLE   70     // Góc vươn gạt hất lon lỗi
#define SERVO_STRIKE_HOLD    450    // Thời gian giữ tay gạt (ms)
#define SERVO_STRIKE_RETURN  400    // Thời gian kéo tay gạt về (ms)
```

---

## 4. QUẢN LÝ CÁC BẢN FIRMWARE

Toàn bộ mã nguồn vi điều khiển được lưu trữ trong thư mục [`esp_firmware/`](../esp_firmware/) chuẩn theo định dạng sketch của Arduino IDE:

1. **`esp_firmware/test1/test1.ino` (Bản chạy chính thức):**
   - Đã tối ưu hóa hoàn chỉnh cho Servo kim loại MG996R.
   - Delay hành trình lon được căn chỉnh chuẩn xác ở **900 ms**.
   - Hỗ trợ giao tiếp TCP Socket ổn định, tự động reconnect WiFi.
2. **`esp_firmware/test1_backup3/test1_backup3.ino` (Bản sao lưu 3):**
   - Lưu trữ dự phòng trạng thái đã hiệu chuẩn thành công với Servo MG996R.
3. **`esp_firmware/test1_backup2/test1_backup2.ino` (Bản sao lưu 2):**
   - Bản sao lưu firmware trước khi tinh chỉnh góc gạt.

---

## 5. HƯỚNG DẪN NẠP CODE QUA ARDUINO IDE

1. Cài đặt board ESP8266 trong Arduino IDE:
   - Thêm URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json` vào mục Preferences.
   - Cài đặt package `esp8266 by ESP8266 Community`.
2. Mở file `esp_firmware/test1/test1.ino`.
3. Cài đặt các thư viện phụ thuộc:
   - `Servo.h` (tích hợp sẵn trong package ESP8266).
4. Thiết lập cấu hình nạp:
   - **Board:** NodeMCU 1.0 (ESP-12E Module)
   - **Upload Speed:** 115200 (hoặc 921600 để nạp nhanh)
   - **CPU Frequency:** 80 MHz hoặc 160 MHz
   - **Port:** Chọn đúng cổng COM của chip nạp nạp (CH340 / CP2102).
5. Bấm nút **Upload**. Mở Serial Monitor (Baudrate **115200**) để lấy địa chỉ IP cấp phát từ WiFi.
