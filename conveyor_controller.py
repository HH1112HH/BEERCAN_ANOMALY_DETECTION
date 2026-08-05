import socket

ESP_HOST = "esp8266-conveyor.local"
ESP_PORT = 12345


class ConveyorController:
    def __init__(self, host=ESP_HOST, port=ESP_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.5)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.5)
            print(f"[OK] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[ERROR] Cannot connect: {e}")
            self.sock = None
            return False

    def send(self, cmd):
        if not self.sock:
            return False
        try:
            self.sock.sendall((cmd + "\n").encode())
            return True
        except:
            self.sock = None
            return False

    def read_line(self, timeout=0.5):
        if not self.sock:
            return None
        try:
            self.sock.settimeout(timeout)
            data = self.sock.recv(1024).decode(errors="ignore")
            if data:
                self.buffer += data
            else:
                self.sock = None
                return None
        except socket.timeout:
            pass
        except:
            self.sock = None
            return None
        if "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            return line.strip()
        return None

    def is_connected(self):
        return self.sock is not None

    def reconnect(self):
        self.disconnect()
        return self.connect()

    def wait_for_inspection(self, timeout=None):
        while True:
            msg = self.read_line(timeout=0.5)
            if msg is None:
                return False
            if "TRIGGER_CAMERA" in msg or "WAITING_INSPECTION" in msg:
                return True
            if timeout is not None:
                timeout -= 0.5
                if timeout <= 0:
                    return False

    def send_ok(self):
        return self.send("OK")

    def send_ng(self):
        return self.send("NG")

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def run_ai_loop(self):
        print("[AI] Waiting for products...")
        while True:
            if not self.wait_for_inspection():
                print("[AI] Connection lost, reconnecting...")
                self.disconnect()
                while not self.connect():
                    import time
                    time.sleep(2)
                continue
            result = input("[AI] OK/NG: ").strip().upper()
            if result == "OK":
                self.send_ok()
                print("PASSED")
            elif result == "NG":
                self.send_ng()
                print("REJECTED")

if __name__ == "__main__":
    ctrl = ConveyorController()
    if ctrl.connect():
        ctrl.run_ai_loop()
