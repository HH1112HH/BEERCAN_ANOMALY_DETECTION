# BeerCan Anomaly Detection

Hệ thống kiểm tra lon gồm backend Python/FastAPI, ESP và giao diện React/Vite có thể deploy riêng.

## Cấu trúc

- `main.py`: camera, phát hiện lon, kiểm tra bất thường và API FastAPI.
- `frontend/`: giao diện React/Vite; không chứa model weights hay thông tin camera.
- `backend_asgi.py`: ASGI entry point có CORS cho frontend chạy khác origin.
- `docs/ANOMALY_MODELS_INTEGRATION.md`: kế hoạch tích hợp DRAEM và SK-RD4AD sau khi train.
- `DESIGN.md`: hướng dẫn giao diện theo Open Design để dùng lại cho các lần thiết kế sau.

## Chạy frontend riêng

```bash
cd frontend
npm ci
npm run dev
```

Vite chạy ở `http://localhost:5173` và proxy API về `http://127.0.0.1:5004`. Muốn trỏ sang API khác, tạo `frontend/.env.local` từ `.env.example` và đặt `VITE_API_BASE_URL`.

Build để deploy static riêng:

```bash
cd frontend
npm run build
```

Deploy thư mục `frontend/dist` lên static host. Backend cần cho phép origin của frontend qua `FRONTEND_ORIGINS` và chạy bằng:

```bash
uvicorn backend_asgi:app --host 0.0.0.0 --port 5004
```

## Nhánh thuật toán

- `algorithm/draem`: tích hợp adapter và checkpoint DRAEM sau khi huấn luyện.
- `algorithm/sk-rd4ad`: tích hợp adapter và checkpoint SK-RD4AD sau khi huấn luyện.

Hiện hai nhánh là nền tảng chung; chưa chứa checkpoint đã train. Xem `docs/ANOMALY_MODELS_INTEGRATION.md` trước khi thêm weights. Không commit dataset, `.env`, credentials hoặc model weights trực tiếp nếu chưa chọn Git LFS/model registry.
