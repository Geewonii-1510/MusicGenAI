import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, request, jsonify, send_from_directory
from music_generator import MusicGenerator
from history_manager import HistoryManager
import os

app = Flask(__name__)

generator = MusicGenerator()
history = HistoryManager()

AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'audio')

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """Nhận prompt → gọi model → trả về thông tin file nhạc."""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    session_id = data.get('session_id')

    if not prompt:
        return jsonify({'error': 'Prompt không được để trống'}), 400

    try:
        # ── Gọi model generate nhạc ──────────────────────────────────────
        # music_generator.py sẽ trả về (audio_filename, duration, metadata)
        result = generator.generate(prompt)
        # ────────────────────────────────────────────────────────────────

        audio_url = f"/static/audio/{result['filename']}"

        # Lưu vào lịch sử
        session_id = history.add_message(
            session_id=session_id,
            prompt=prompt,
            audio_url=audio_url,
            duration=result.get('duration', 0),
            metadata=result.get('metadata', {})
        )

        return jsonify({
            'session_id': session_id,
            'audio_url': audio_url,
            'duration': result.get('duration', 0),
            'metadata': result.get('metadata', {}),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Trả về danh sách tất cả sessions."""
    return jsonify(history.get_all_sessions())


@app.route('/api/history/<session_id>', methods=['GET'])
def get_session(session_id):
    """Trả về chi tiết một session."""
    session = history.get_session(session_id)
    if not session:
        return jsonify({'error': 'Không tìm thấy session'}), 404
    return jsonify(session)


@app.route('/api/history/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Xóa một session."""
    history.delete_session(session_id)
    return jsonify({'status': 'deleted'})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Xóa toàn bộ lịch sử."""
    history.clear_all()
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    print("=" * 50)
    print("  MusicGenAI Web Interface")
    print("  Mở trình duyệt tại: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
