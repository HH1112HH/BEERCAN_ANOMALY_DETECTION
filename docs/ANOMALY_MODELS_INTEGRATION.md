# DRAEM MODEL INTEGRATION & INFERENCE SPECIFICATION

Tài liệu kỹ thuật đặc tả việc tích hợp mô hình học sâu DRAEM (Discriminative Reconstructive Anomaly Embedding) vào pipeline kiểm tra thời gian thực trong [`src/draem_model.py`](../src/draem_model.py).

---

## 1. TỔNG QUAN TÍCH HỢP

Mô hình DRAEM thay thế hoàn toàn các phương pháp trích xuất đặc trưng truyền thống nhờ khả năng:
1. Tự động khoanh vùng bất thường chi tiết theo từng pixel mà không cần gán nhãn pixel-level trong tập huấn luyện.
2. Sinh bản đồ nhiệt lỗi (Heatmap) trực quan phục vụ công nhân và kỹ sư QC giám sát.
3. Xuất điểm số định lượng duy nhất (`score` từ `0.000` đến `1.000`) để phân loại `GOOD` / `NG`.

---

## 2. GIAO DIỆN HÀM SUY LUẬN (INFERENCE CONTRACT)

Module [`src/draem_model.py`](../src/draem_model.py) cung cấp lớp `DRAEM` chuẩn:

```python
from src.draem_model import DRAEM

# Khởi tạo mô hình trên GPU
model = DRAEM(
    model_path="models/draem.pth",
    threshold=0.25,
    device="cuda:0"
)

# Suy luận trên ảnh đáy lon đã cắt tròn (numpy array BGR kích thước 256x256)
result = model.predict(image_bgr)
```

### Cấu trúc kết quả trả về (`result` dictionary):

```python
{
    "score": 0.7852,             # float: Điểm bất thường từ 0.0 đến 1.0
    "is_anomaly": True,          # bool: True nếu score >= threshold
    "threshold": 0.25,           # float: Ngưỡng quyết định hiện tại
    "heatmap": np.ndarray,       # ndarray (256, 256, 3) BGR: Bản đồ nhiệt màu JET
    "anomaly_map": np.ndarray,   # ndarray (256, 256) float32: Ma trận xác suất chuẩn hóa [0, 1]
    "recon_image": np.ndarray,   # ndarray (256, 256, 3) BGR: Ảnh đáy lon đã được tái tạo sạch
    "details": {
        "device": "cuda:0",
        "top_k_pixels": 256,
        "inference_time_ms": 14.2
    }
}
```

---

## 3. QUY TRÌNH BIẾN ĐỔI TENSOR & TIỀN XỬ LÝ (PREPROCESSING)

1. **Chuẩn hóa kích thước:** Ảnh đáy lon đầu vào được nội suy (Bilinear Interpolation) về đúng độ phân giải `(256, 256)`.
2. **Chuyển đổi không gian màu:** BGR $\to$ RGB $\to$ Chuyển thành PyTorch Tensor Float32.
3. **Chuẩn hóa cường độ sáng (Normalization):**
   $$x_{norm} = \frac{x}{255.0}$$
   Áp dụng chuẩn hóa ImageNet theo từng kênh:
   $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
4. **Nạp Tensor lên GPU:** Chuyển tensor vào bộ nhớ VRAM của GPU (`cuda:0`), thêm chiều Batch `(1, 3, 256, 256)`.

---

## 4. KIẾN TRÚC MẠNG NƠ-RON

Mô hình bao gồm 2 mạng con:

```text
Input Image (3 channels) ───> [ Reconstructive Sub-Network ] ───> Reconstructed Image (3 channels)
           │                                                                    │
           └──────────────────────────┬─────────────────────────────────────────┘
                                      ▼ Concatenate (6 channels)
                      [ Discriminative Sub-Network (U-Net) ]
                                      │
                                      ▼
                      Anomaly Probability Map (256x256)
```

1. **Reconstructive Sub-Network:**
   - Kiến trúc Encoder-Decoder với 6 khối tích chập tích hợp Skip Connections.
   - Hàm kích hoạt: LeakyReLU với độ dốc âm $0.1$.
2. **Discriminative Sub-Network:**
   - Nhận đầu vào ghép nối 6 kênh: Ảnh gốc 3 kênh + Ảnh tái tạo 3 kênh.
   - Đầu ra: Bản đồ dự đoán 1 kênh với hàm kích hoạt Sigmoid đưa giá trị từng pixel về đoạn $[0, 1]$.

---

## 5. TẠO HEATMAP MÀU JET ĐỒNG BỘ TRÊN GIAO DIỆN

Để người vận hành dễ quan sát vị trí dị tật trên Web Dashboard:
1. Lấy bản đồ dự đoán ma trận $M_{pred} \in [0, 1]$.
2. Nhân với 255 và ép kiểu sang `uint8`.
3. Áp dụng bảng màu `cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)`.
4. Trộn (Blend) theo tỷ lệ 60% ảnh gốc + 40% ảnh Heatmap:
   $$\text{Overlay} = 0.6 \times I_{raw} + 0.4 \times \text{Heatmap}_{JET}$$
5. Lưu kết quả vào thư mục `captures/` và truyền trực tiếp qua WebSocket đến Web Dashboard.
