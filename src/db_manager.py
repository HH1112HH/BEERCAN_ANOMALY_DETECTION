import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("db_manager")

DB_DIR = Path("results")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "inspection_database.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    target_count INTEGER DEFAULT 100,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    can_number INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    score REAL,
                    defects TEXT,
                    details TEXT,
                    image_file TEXT,
                    FOREIGN KEY (session_id) REFERENCES test_sessions(id)
                )
            """)
            # Ensure image_file column exists in existing DB
            cur = conn.cursor()
            cols = [c[1] for c in cur.execute("PRAGMA table_info(inspection_records)").fetchall()]
            if "image_file" not in cols:
                conn.execute("ALTER TABLE inspection_records ADD COLUMN image_file TEXT")
        logger.info(f"Database initialized at {DB_PATH}")
    finally:
        conn.close()


def get_active_session():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM test_sessions
            WHERE status = 'active'
            ORDER BY id DESC LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def create_new_session(name=None, target_count=100):
    conn = get_connection()
    try:
        with conn:
            # Mark existing active sessions as completed
            conn.execute("UPDATE test_sessions SET status = 'completed' WHERE status = 'active'")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not name:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) as cnt FROM test_sessions")
                cnt = cur.fetchone()["cnt"] + 1
                name = f"Đợt test #{cnt} ({target_count} lon)"
            cur = conn.execute(
                "INSERT INTO test_sessions (name, target_count, created_at, status) VALUES (?, ?, ?, 'active')",
                (name, int(target_count), now)
            )
            session_id = cur.lastrowid
        return {
            "id": session_id,
            "name": name,
            "target_count": int(target_count),
            "created_at": now,
            "status": "active"
        }
    finally:
        conn.close()


def get_or_create_active_session(default_target=100):
    sess = get_active_session()
    if not sess:
        sess = create_new_session(target_count=default_target)
    return sess


def save_record(session_id, can_number, timestamp, verdict, score, defects, details, image_file=None):
    conn = get_connection()
    try:
        defects_str = json.dumps(defects) if isinstance(defects, (list, dict)) else str(defects)
        details_str = json.dumps(details) if isinstance(details, (list, dict)) else str(details)
        with conn:
            conn.execute("""
                INSERT INTO inspection_records 
                (session_id, can_number, timestamp, verdict, score, defects, details, image_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, can_number, timestamp, verdict, float(score) if score is not None else 0.0, defects_str, details_str, image_file))
    except Exception as e:
        logger.error(f"Error saving record to DB: {e}")
    finally:
        conn.close()


def get_session_records(session_id, limit=100):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, can_number as id_num, timestamp, verdict, score, defects, details, image_file
            FROM inspection_records
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        rows = cur.fetchall()
        result = []
        for r in rows:
            try:
                defects = json.loads(r["defects"]) if r["defects"] else []
            except Exception:
                defects = [r["defects"]] if r["defects"] else []
            
            img_file = r["image_file"]
            if not img_file:
                # Auto check if capture file exists on disk
                pattern = f"Lon_{r['id_num']:03d}_*.jpg"
                found = list(Path("captures").glob(pattern))
                if found:
                    img_file = found[0].name

            result.append({
                "id": r["id_num"],
                "timestamp": r["timestamp"],
                "verdict": r["verdict"],
                "score": round(r["score"], 4) if r["score"] is not None else 0.0,
                "defects": defects,
                "image_file": img_file,
            })
        return result
    finally:
        conn.close()


def get_session_stats(session_id):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN verdict = 'GOOD' THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN verdict = 'NG' THEN 1 ELSE 0 END) as ng,
                MAX(can_number) as last_can_number
            FROM inspection_records
            WHERE session_id = ?
        """, (session_id,))
        row = cur.fetchone()
        total = row["total"] or 0
        good = row["good"] or 0
        ng = row["ng"] or 0
        last_can = row["last_can_number"] or 0
        rate = round(good / total * 100, 1) if total > 0 else 100.0
        return {
            "total": total,
            "good": good,
            "ng": ng,
            "rate": rate,
            "last_can_number": last_can,
        }
    finally:
        conn.close()


def delete_last_record(session_id):
    conn = get_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, can_number, verdict FROM inspection_records
                WHERE session_id = ?
                ORDER BY can_number DESC LIMIT 1
            """, (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM inspection_records WHERE id = ?", (row["id"],))
            return dict(row)
    finally:
        conn.close()


def export_session_csv_text(session_id):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, created_at FROM test_sessions WHERE id = ?", (session_id,))
        sess = cur.fetchone()
        sess_name = sess["name"] if sess else f"Session_{session_id}"
        
        cur.execute("""
            SELECT can_number, timestamp, verdict, score, defects, image_file
            FROM inspection_records
            WHERE session_id = ?
            ORDER BY can_number ASC
        """, (session_id,))
        rows = cur.fetchall()

        lines = ["Lon STT,Thời Gian,Kết Quả (Verdict),Điểm Lỗi (Score),Khuyết Tật (Defects),File Ảnh Phân Tích"]
        for r in rows:
            defects = r["defects"] or "-"
            img = r["image_file"] or "-"
            lines.append(f"{r['can_number']},{r['timestamp']},{r['verdict']},{r['score']:.4f},\"{defects}\",{img}")
        return "\n".join(lines)
    finally:
        conn.close()

