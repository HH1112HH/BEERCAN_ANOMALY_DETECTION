# Hướng dẫn chạy hệ thống

## Bước 1: Kích hoạt môi trường

```bash
source venv/bin/activate
```

## Bước 2: Cài thư viện

```bash
pip install -r requirements.txt
```

## Bước 3: Kiểm tra camera trong config.yaml

File `config.yaml` đã có sẵn thông tin camera. Nếu cần sửa IP/user/pass thì mở file ra sửa.

## Bước 4: Chạy hệ thống

```bash
python3 main.py
```

## Bước 5: Mở trình duyệt

Vào `http://localhost:5000` để xem dashboard.

---

### Nếu bị lỗi "camera connection"

Kiểm tra lại `rtsp_url` trong `config.yaml` cho đúng với camera thực tế.

### Nếu muốn dừng

Nhấn `Ctrl+C` trong terminal.
