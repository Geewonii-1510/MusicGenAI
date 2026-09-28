"""
music_generator.py

Wrapper tích hợp TextToMusicTransformer (model.py) với Flask web server.

Luồng generate:
    prompt (str)
      ↓
    tokenize text → text_ids  (dùng text vocab)
      ↓
    Encoder → memory
      ↓
    Decoder (autoregressive) → music token ids
      ↓
    decode tokens → MIDI notes
      ↓
    convert MIDI → WAV          ← cần FluidSynth hoặc pretty_midi
      ↓
    lưu vào static/audio/xxx.wav
      ↓
    trả về filename cho Flask

═══════════════════════════════════════════════════════════════
  ĐỂ TÍCH HỢP MODEL THẬT, bạn cần cung cấp:
    1. checkpoints/best_model.pt     (checkpoint đã train)
    2. vocab/text_vocab.json         ({"<PAD>":0, "một":5, ...})
    3. vocab/music_vocab.json        ({"<PAD>":0, "NOTE_ON_60":3, ...})
    4. (Tuỳ chọn) soundfont .sf2     (để convert MIDI→WAV)
═══════════════════════════════════════════════════════════════
"""

import os
import json
import time
import uuid
import random

# torch và model được import lazy (chỉ khi có model thật)
# → Flask chạy được ở chế độ demo dù chưa cài PyTorch
_torch = None
_model_consts = {}

def _import_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch

def _get_model_consts():
    global _model_consts
    if not _model_consts:
        from model import (
            TextToMusicTransformer,
            MUSIC_BOS_ID, MUSIC_EOS_ID, MUSIC_PAD_ID,
            TEXT_PAD_ID, MAX_TEXT_LENGTH, MAX_MUSIC_LENGTH,
        )
        _model_consts = dict(
            TextToMusicTransformer=TextToMusicTransformer,
            MUSIC_BOS_ID=MUSIC_BOS_ID,
            MUSIC_EOS_ID=MUSIC_EOS_ID,
            MUSIC_PAD_ID=MUSIC_PAD_ID,
            TEXT_PAD_ID=TEXT_PAD_ID,
            MAX_TEXT_LENGTH=MAX_TEXT_LENGTH,
            MAX_MUSIC_LENGTH=MAX_MUSIC_LENGTH,
        )
    return _model_consts

# ── Đường dẫn ──────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
AUDIO_FOLDER  = os.path.join(BASE_DIR, 'static', 'audio')
CHECKPOINT    = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pt')
TEXT_VOCAB_F  = os.path.join(BASE_DIR, 'vocab', 'text_vocab.json')
MUSIC_VOCAB_F = os.path.join(BASE_DIR, 'vocab', 'music_vocab.json')


# ══════════════════════════════════════════════════════════════════
class MusicGenerator:
    """
    Wrapper chính. Flask gọi generator.generate(prompt) để lấy file nhạc.
    """

    def __init__(self):
        os.makedirs(AUDIO_FOLDER, exist_ok=True)

        # Kiểm tra xem có đủ file model chưa
        self._ready = (
            os.path.exists(CHECKPOINT) and
            os.path.exists(TEXT_VOCAB_F) and
            os.path.exists(MUSIC_VOCAB_F)
        )

        if self._ready:
            self._load_real_model()
        else:
            missing = []
            if not os.path.exists(CHECKPOINT):    missing.append('checkpoints/best_model.pt')
            if not os.path.exists(TEXT_VOCAB_F):  missing.append('vocab/text_vocab.json')
            if not os.path.exists(MUSIC_VOCAB_F): missing.append('vocab/music_vocab.json')
            print(f"[MusicGenerator] DEMO mode (thieu file: {', '.join(missing)})")

    # ── Load model thật ──────────────────────────────────────────────
    def _load_real_model(self):
        """Load vocab + model từ checkpoint."""
        torch = _import_torch()
        consts = _get_model_consts()

        print("[MusicGenerator] Loading vocab...")
        with open(TEXT_VOCAB_F,  'r', encoding='utf-8') as f:
            self.text_vocab  = json.load(f)
        with open(MUSIC_VOCAB_F, 'r', encoding='utf-8') as f:
            self.music_vocab = json.load(f)

        self.id_to_music = {v: k for k, v in self.music_vocab.items()}

        text_vocab_size  = len(self.text_vocab)
        music_vocab_size = len(self.music_vocab)
        print(f"[MusicGenerator] Text vocab: {text_vocab_size} | Music vocab: {music_vocab_size}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[MusicGenerator] Device: {self.device}")

        print("[MusicGenerator] Loading model checkpoint...")
        TextToMusicTransformer = consts['TextToMusicTransformer']
        self.model = TextToMusicTransformer(
            text_vocab_size=text_vocab_size,
            music_vocab_size=music_vocab_size,
        ).to(self.device)

        checkpoint = torch.load(CHECKPOINT, map_location=self.device)
        state_dict = checkpoint.get('model', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[MusicGenerator] Model ready!")

    # ════════════════════════════════════════════════════════════════
    def generate(self, prompt: str) -> dict:
        """
        Tạo nhạc từ prompt. Được gọi bởi Flask route /api/generate.

        Returns
        -------
        dict:
            filename  (str)   - tên file WAV trong static/audio/
            duration  (float) - thời lượng ước tính (giây)
            metadata  (dict)  - thông tin thêm
        """
        if self._ready:
            return self._real_generate(prompt)
        else:
            return self._demo_generate(prompt)

    # ── Chế độ DEMO (chưa có model) ─────────────────────────────────
    def _demo_generate(self, prompt: str) -> dict:
        """
        Giả lập generate để test giao diện.
        Không tạo file âm thanh thật.
        """
        time.sleep(random.uniform(1.5, 3.0))
        filename = f"demo_{uuid.uuid4().hex[:8]}.wav"
        return {
            'filename': filename,
            'duration': random.uniform(15, 45),
            'metadata': {
                'mode':         'demo',
                'device':       'N/A',
                'prompt_words': len(prompt.split()),
                'note':         'Copy checkpoints/ va vocab/ vao thu muc du an de dung model that.'
            }
        }

    # ── Generate THẬT ────────────────────────────────────────────────
    def _real_generate(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> dict:
        torch = _import_torch()
        consts = _get_model_consts()
        MAX_MUSIC_LENGTH = consts['MAX_MUSIC_LENGTH']
        MUSIC_BOS_ID     = consts['MUSIC_BOS_ID']
        MUSIC_EOS_ID     = consts['MUSIC_EOS_ID']
        MUSIC_PAD_ID     = consts['MUSIC_PAD_ID']
        if max_length is None:
            max_length = MAX_MUSIC_LENGTH

        with torch.no_grad():

            # ── Bước 1: Tokenize văn bản ────────────────────────────
            text_ids = self._tokenize(prompt)          # [1, text_len]
            text_ids = text_ids.to(self.device)

            # ── Bước 2: Encode ──────────────────────────────────────
            memory, text_pad_mask = self.model.encode(text_ids)

            # ── Bước 3: Decode autoregressive ───────────────────────
            generated = [MUSIC_BOS_ID]

            for _ in range(max_length - 1):
                music_ids = torch.tensor(
                    [generated], dtype=torch.long, device=self.device
                )

                logits = self.model.decode(
                    music_ids=music_ids,
                    memory=memory,
                    text_padding_mask=text_pad_mask,
                )  # [1, cur_len, vocab_size]

                next_logits = logits[0, -1, :] / temperature
                next_token = self._top_k_sample(next_logits, top_k)
                generated.append(next_token)

                if next_token == MUSIC_EOS_ID:
                    break

            # ── Bước 4: Decode token ids → MIDI ─────────────────────
            token_names = [
                self.id_to_music.get(tid, '<UNK>')
                for tid in generated[1:]
                if tid not in (MUSIC_EOS_ID, MUSIC_PAD_ID)
            ]
            midi_path = self._tokens_to_midi(token_names)

            # ── Bước 5: Convert MIDI → WAV ───────────────────────────
            filename  = f"music_{uuid.uuid4().hex[:8]}.wav"
            wav_path  = os.path.join(AUDIO_FOLDER, filename)
            duration  = self._midi_to_wav(midi_path, wav_path)

        return {
            'filename': filename,
            'duration': duration,
            'metadata': {
                'mode':        'model',
                'device':      str(self.device),
                'tokens':      len(generated),
                'text_vocab':  len(self.text_vocab),
                'music_vocab': len(self.music_vocab),
            }
        }

    # ── Tokenize ────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> 'torch.Tensor':
        """Chuyển chuỗi text → tensor token ids."""
        torch  = _import_torch()
        consts = _get_model_consts()
        TEXT_PAD_ID    = consts['TEXT_PAD_ID']
        MAX_TEXT_LENGTH = consts['MAX_TEXT_LENGTH']

        UNK = self.text_vocab.get('<UNK>', 1)
        tokens = [
            self.text_vocab.get(w.lower(), UNK)
            for w in text.strip().split()
        ]

        tokens = tokens[:MAX_TEXT_LENGTH]
        tokens += [TEXT_PAD_ID] * (MAX_TEXT_LENGTH - len(tokens))

        return torch.tensor([tokens], dtype=torch.long)   # [1, MAX_TEXT_LENGTH]

    # ── Top-k sampling ───────────────────────────────────────────────
    @staticmethod
    def _top_k_sample(logits: 'torch.Tensor', k: int) -> int:
        """Lấy mẫu từ top-k token có xác suất cao nhất."""
        torch = _import_torch()
        if k > 0:
            values, _ = torch.topk(logits, k)
            threshold  = values[-1]
            logits     = logits.masked_fill(logits < threshold, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ── Token → MIDI ─────────────────────────────────────────────────
    def _tokens_to_midi(self, token_names: list[str]) -> str:
        """
        Chuyển danh sách music token (dạng tên) → file MIDI.

        Token format (theo model của bạn):
            NOTE_ON_<pitch>      → bật nốt nhạc
            NOTE_OFF_<pitch>     → tắt nốt nhạc (nếu có)
            VELOCITY_<value>     → cường độ (0–127)
            DURATION_<steps>     → thời lượng
            TIME_SHIFT_<steps>   → khoảng cách thời gian

        ══════════════════════════════════════════════════════════════
        Cần cài: pip install pretty_midi
        ══════════════════════════════════════════════════════════════
        """
        try:
            import pretty_midi
        except ImportError:
            raise RuntimeError(
                "Chua cai pretty_midi. Chay: pip install pretty_midi"
            )

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Grand Piano

        # Giá trị mặc định
        current_time = 0.0
        velocity     = 80
        TIME_STEP    = 0.125   # 1 step = 0.125 giây (tuỳ chỉnh nếu cần)

        pending_notes = {}   # pitch → start_time

        for token in token_names:
            try:
                if token.startswith('NOTE_ON_'):
                    pitch = int(token.split('_')[-1])
                    pending_notes[pitch] = (current_time, velocity)

                elif token.startswith('NOTE_OFF_'):
                    pitch = int(token.split('_')[-1])
                    if pitch in pending_notes:
                        start, vel = pending_notes.pop(pitch)
                        note = pretty_midi.Note(
                            velocity=vel,
                            pitch=pitch,
                            start=start,
                            end=max(current_time, start + TIME_STEP),
                        )
                        instrument.notes.append(note)

                elif token.startswith('VELOCITY_'):
                    velocity = min(127, max(1, int(token.split('_')[-1])))

                elif token.startswith('DURATION_'):
                    steps = int(token.split('_')[-1])
                    # Đóng tất cả nốt đang mở sau `steps` bước
                    end_time = current_time + steps * TIME_STEP
                    for pitch, (start, vel) in list(pending_notes.items()):
                        note = pretty_midi.Note(
                            velocity=vel,
                            pitch=pitch,
                            start=start,
                            end=end_time,
                        )
                        instrument.notes.append(note)
                    pending_notes.clear()
                    current_time = end_time

                elif token.startswith('TIME_SHIFT_'):
                    steps = int(token.split('_')[-1])
                    current_time += steps * TIME_STEP

            except (ValueError, IndexError):
                continue  # bỏ qua token lỗi

        # Đóng các nốt còn đang mở
        for pitch, (start, vel) in pending_notes.items():
            note = pretty_midi.Note(
                velocity=vel, pitch=pitch,
                start=start, end=max(current_time, start + TIME_STEP)
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        # Lưu file MIDI tạm
        midi_path = os.path.join(AUDIO_FOLDER, f"tmp_{uuid.uuid4().hex[:8]}.mid")
        midi.write(midi_path)
        return midi_path

    # ── MIDI → WAV ───────────────────────────────────────────────────
    def _midi_to_wav(self, midi_path: str, wav_path: str) -> float:
        """
        Convert MIDI → WAV dùng pretty_midi + FluidSynth.

        Cần cài:
            pip install pretty_midi
            conda install -c conda-forge fluidsynth   (hoặc cài binary)

        Nếu không có FluidSynth, file MIDI được giữ nguyên (không có WAV).

        ══════════════════════════════════════════════════════════════
        SOUNDFONT: Đặt file .sf2 vào thư mục dự án, đặt tên
                   'soundfont.sf2' hoặc sửa đường dẫn bên dưới.
        ══════════════════════════════════════════════════════════════
        """
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
            duration = midi.get_end_time()

            # Tìm soundfont
            sf2_candidates = [
                os.path.join(BASE_DIR, 'soundfont.sf2'),
                r'C:\soundfonts\GeneralUser.sf2',         # Windows default
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',  # Linux default
            ]
            sf2 = next((p for p in sf2_candidates if os.path.exists(p)), None)

            if sf2:
                audio = midi.fluidsynth(fs=44100, sf2_path=sf2)
                import scipy.io.wavfile as wav
                wav.write(wav_path, 44100, audio)
            else:
                # Không có FluidSynth/soundfont → copy MIDI thay thế
                import shutil
                shutil.copy(midi_path, wav_path.replace('.wav', '.mid'))
                wav_path = wav_path.replace('.wav', '.mid')
                print("[MusicGenerator] Khong co FluidSynth/soundfont. Luu file MIDI.")

            # Xóa file MIDI tạm
            try:
                os.remove(midi_path)
            except OSError:
                pass

            return duration

        except Exception as e:
            print(f"[MusicGenerator] Loi convert MIDI->WAV: {e}")
            return 0.0
