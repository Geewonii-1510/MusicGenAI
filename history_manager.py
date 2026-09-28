import json
import uuid
import os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'chat_history.json')


class HistoryManager:
    def __init__(self):
        self._ensure_file()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ensure_file(self):
        if not os.path.exists(HISTORY_FILE):
            self._save({})

    def _load(self) -> dict:
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: dict):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Public API ───────────────────────────────────────────────────────

    def add_message(self, session_id, prompt, audio_url, duration=0, metadata=None) -> str:
        """Thêm một cặp (prompt → nhạc) vào session. Tạo session mới nếu chưa có."""
        data = self._load()

        if not session_id or session_id not in data:
            session_id = str(uuid.uuid4())
            # Lấy 6 từ đầu của prompt làm title
            title = ' '.join(prompt.split()[:6])
            if len(prompt.split()) > 6:
                title += '...'
            data[session_id] = {
                'id': session_id,
                'title': title,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'messages': []
            }

        message = {
            'id': str(uuid.uuid4()),
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        }
        response = {
            'id': str(uuid.uuid4()),
            'role': 'assistant',
            'audio_url': audio_url,
            'duration': duration,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }

        data[session_id]['messages'].extend([message, response])
        data[session_id]['updated_at'] = datetime.now().isoformat()
        self._save(data)
        return session_id

    def get_all_sessions(self) -> list:
        """Trả về danh sách sessions, sắp xếp mới nhất trước."""
        data = self._load()
        sessions = list(data.values())
        sessions.sort(key=lambda s: s.get('updated_at', ''), reverse=True)
        # Chỉ trả về metadata (không kèm messages đầy đủ)
        return [
            {
                'id': s['id'],
                'title': s['title'],
                'created_at': s['created_at'],
                'updated_at': s['updated_at'],
                'message_count': len(s['messages']) // 2
            }
            for s in sessions
        ]

    def get_session(self, session_id: str) -> dict | None:
        data = self._load()
        return data.get(session_id)

    def delete_session(self, session_id: str):
        data = self._load()
        data.pop(session_id, None)
        self._save(data)

    def clear_all(self):
        self._save({})
