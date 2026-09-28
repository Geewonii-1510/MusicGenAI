# MusicGenAI – Web Interface 🎵

Giao diện web theo phong cách **ChatGPT** cho mô hình AI sinh nhạc **MusicGenAI** (Text-to-Music Transformer).

Nhập mô tả bằng tiếng Việt hoặc tiếng Anh → AI tự động tạo ra bản nhạc.

---

## ✨ Tính năng

- 💬 Chat interface nhập prompt như ChatGPT
- 🎵 Audio player nghe nhạc trực tiếp trong trình duyệt
- 📋 Sidebar lịch sử hội thoại (lưu tất cả session)
- 🔍 Tìm kiếm lịch sử
- ✨ Welcome screen với ví dụ gợi ý
- ⏳ Loading animation khi model đang generate
- 🌙 Dark theme màu tím/navy đẹp
- 📱 Responsive (desktop & mobile)

---

## 🚀 Cài đặt & Chạy

### 1. Clone repo

```bash
git clone https://github.com/geewoni-1510/MusicGenAI-UI.git
cd MusicGenAI-UI
```

### 2. Cài dependencies

```bash
pip install flask pretty_midi
```

> Nếu muốn convert MIDI → WAV: cài thêm FluidSynth + soundfont `.sf2`

### 3. (Tuỳ chọn) Thêm model thật

Đặt các file sau vào thư mục dự án:

```
MusicGenAI-UI/
├── checkpoints/
│   └── best_model.pt       ← checkpoint đã train
└── vocab/
    ├── text_vocab.json      ← {"<PAD>":0, "một":5, ...}
    └── music_vocab.json     ← {"NOTE_ON_60":3, ...}
```

Nếu không có → chạy ở **chế độ demo** (test giao diện).

### 4. Chạy server

```bash
python app.py
# → Mở trình duyệt: http://localhost:5000
```

---

## 🏗️ Kiến trúc Model (model.py)

```
Text prompt
    ↓
Text Embedding + Positional Encoding
    ↓
Transformer Encoder (4 layers, d_model=256, 8 heads)
    ↓
Memory
    ↓
Transformer Decoder (4 layers, autoregressive)
    ↓
Music Token Logits → Top-k Sampling
    ↓
NOTE_ON / VELOCITY / TIME_SHIFT / DURATION tokens
    ↓
MIDI → WAV
```

---

## 📁 Cấu trúc dự án

```
MusicGenAI-UI/
├── app.py                  # Flask server
├── model.py                # TextToMusicTransformer
├── music_generator.py      # Inference pipeline
├── history_manager.py      # Lưu lịch sử chat
├── requirements.txt
├── templates/
│   └── index.html          # Giao diện chính
└── static/
    ├── css/style.css
    ├── js/app.js
    └── audio/              # File nhạc generated (gitignored)
```

---

## 📦 Dependencies

- Python 3.10+
- Flask 3.x
- PyTorch (để chạy model thật)
- pretty_midi (để tạo MIDI)
- FluidSynth + soundfont (để convert MIDI → WAV, tuỳ chọn)
