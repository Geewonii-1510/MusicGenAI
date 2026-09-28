"""
model.py

Định nghĩa mô hình Transformer cho bài toán Text-to-Music.

Input:
    Một chuỗi token văn bản, ví dụ:
        "một bản nhạc vui vẻ với piano"

Output:
    Một chuỗi Music Token:
        BOS
        NOTE_ON_60
        VELOCITY_90
        DURATION_4
        TIME_SHIFT_2
        ...
        EOS

Kiến trúc:

    Text
      ↓
    Text Embedding
      ↓
    Positional Encoding
      ↓
    Transformer Encoder
      ↓
    Memory
      ↓
    Transformer Decoder
      ↓
    Linear Projection
      ↓
    Music Token probabilities
"""

import math

import torch
import torch.nn as nn


# ============================================================
# 1. CẤU HÌNH MẶC ĐỊNH
# ============================================================

D_MODEL = 256
N_HEADS = 8

NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

MAX_TEXT_LENGTH = 64
MAX_MUSIC_LENGTH = 512

# Text vocabulary
TEXT_PAD_ID = 0

# Music vocabulary
MUSIC_PAD_ID = 0
MUSIC_BOS_ID = 1
MUSIC_EOS_ID = 2


# ============================================================
# 2. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding cho Transformer.

    Transformer không tự biết thứ tự của token.
    Positional Encoding bổ sung thông tin vị trí:

        token 1 → position 1
        token 2 → position 2
        token 3 → position 3
        ...

    Parameters
    ----------
    d_model : int
        Kích thước embedding.

    max_len : int
        Số lượng vị trí tối đa.

    dropout : float
        Tỷ lệ dropout.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Tensor lưu positional encoding.
        pe = torch.zeros(max_len, d_model)

        # Vị trí token.
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        # Công thức positional encoding.
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer:
        # - không phải parameter cần train
        # - nhưng vẫn được lưu trong model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                [batch_size, sequence_length, d_model]

        Returns
        -------
        torch.Tensor
            Tensor sau khi cộng positional encoding.
        """

        sequence_length = x.size(1)

        x = x + self.pe[:, :sequence_length]

        return self.dropout(x)


# ============================================================
# 3. TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):
    """
    Encoder xử lý chuỗi text token.

    Ví dụ:

        "một bản nhạc vui vẻ"

              ↓

        [BOS, một, bản, nhạc, vui, vẻ, EOS]

              ↓

        Embedding

              ↓

        Positional Encoding

              ↓

        Transformer Encoder

              ↓

        Text representation
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        num_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_length: int = MAX_TEXT_LENGTH,
        pad_id: int = TEXT_PAD_ID,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Text token embedding.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        # Positional Encoding.
        self.position = PositionalEncoding(
            d_model=d_model,
            max_len=max_length,
            dropout=dropout,
        )

        # Một Transformer Encoder Layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        # Nhiều Encoder Layer.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor

            Shape:

                [batch_size, text_length]

        Returns
        -------
        torch.Tensor

            Shape:

                [batch_size, text_length, d_model]
        """

        # Padding mask.
        padding_mask = input_ids.eq(self.pad_id)

        # Embedding.
        x = self.embedding(input_ids)

        # Scale embedding.
        x = x * math.sqrt(self.d_model)

        # Positional Encoding.
        x = self.position(x)

        # Transformer Encoder.
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        return x


# ============================================================
# 4. MUSIC DECODER
# ============================================================

class MusicDecoder(nn.Module):
    """
    Decoder sinh chuỗi Music Token.

    Ví dụ:

        BOS
         ↓
        NOTE_ON_60
         ↓
        VELOCITY_90
         ↓
        DURATION_4
         ↓
        TIME_SHIFT_2
         ↓
        ...
         ↓
        EOS

    Decoder sử dụng causal mask để đảm bảo model
    không nhìn thấy token tương lai trong quá trình training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        num_layers: int = NUM_DECODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_length: int = MAX_MUSIC_LENGTH,
        pad_id: int = MUSIC_PAD_ID,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Music token embedding.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        # Positional Encoding.
        self.position = PositionalEncoding(
            d_model=d_model,
            max_len=max_length,
            dropout=dropout,
        )

        # Decoder Layer.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        # Transformer Decoder.
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # Chuyển hidden state → music token logits.
        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
        )

    def generate_causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Tạo causal mask.

        Token hiện tại chỉ được nhìn thấy:
            token hiện tại
            token phía trước

        Không được nhìn thấy:
            token tương lai.

        Returns
        -------
        torch.Tensor
            Shape:
                [sequence_length, sequence_length]
        """

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return mask

    def forward(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        target_ids : torch.Tensor
            Music token IDs.

            Shape:
                [batch_size, music_length]

        memory : torch.Tensor
            Output từ Text Encoder.

            Shape:
                [batch_size, text_length, d_model]

        memory_padding_mask : torch.Tensor | None
            Padding mask của text.

        Returns
        -------
        torch.Tensor

            Music token logits.

            Shape:
                [batch_size, music_length, music_vocab_size]
        """

        batch_size, music_length = target_ids.shape

        # Padding mask cho music.
        target_padding_mask = target_ids.eq(self.pad_id)

        # Causal mask.
        causal_mask = self.generate_causal_mask(
            sequence_length=music_length,
            device=target_ids.device,
        )

        # Music embedding.
        x = self.embedding(target_ids)

        # Scale embedding.
        x = x * math.sqrt(self.d_model)

        # Positional Encoding.
        x = self.position(x)

        # Transformer Decoder.
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        x = self.norm(x)

        # Project thành logits.
        logits = self.output_projection(x)

        return logits


# ============================================================
# 5. TEXT → MUSIC TRANSFORMER
# ============================================================

class TextToMusicTransformer(nn.Module):
    """
    Mô hình hoàn chỉnh Text-to-Music.

    Kiến trúc:

        Text IDs
            │
            ▼
        Text Encoder
            │
            ▼
        Text Memory
            │
            ▼
        Music Decoder
            │
            ▼
        Music Token Logits

    Parameters
    ----------
    text_vocab_size : int
        Kích thước text vocabulary.

    music_vocab_size : int
        Kích thước music vocabulary.
    """

    def __init__(
        self,
        text_vocab_size: int,
        music_vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        num_decoder_layers: int = NUM_DECODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_text_length: int = MAX_TEXT_LENGTH,
        max_music_length: int = MAX_MUSIC_LENGTH,
        text_pad_id: int = TEXT_PAD_ID,
        music_pad_id: int = MUSIC_PAD_ID,
    ):
        super().__init__()

        self.text_vocab_size = text_vocab_size
        self.music_vocab_size = music_vocab_size

        self.text_pad_id = text_pad_id
        self.music_pad_id = music_pad_id

        self.encoder = TextEncoder(
            vocab_size=text_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_text_length,
            pad_id=text_pad_id,
        )

        self.decoder = MusicDecoder(
            vocab_size=music_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_music_length,
            pad_id=music_pad_id,
        )

    def encode(
        self,
        text_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text thành memory.

        Returns
        -------
        memory :
            [batch_size, text_length, d_model]

        text_padding_mask :
            [batch_size, text_length]
        """

        text_padding_mask = text_ids.eq(
            self.text_pad_id
        )

        memory = self.encoder(text_ids)

        return memory, text_padding_mask

    def decode(
        self,
        music_ids: torch.Tensor,
        memory: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode music tokens.
        """

        return self.decoder(
            target_ids=music_ids,
            memory=memory,
            memory_padding_mask=text_padding_mask,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        music_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass hoàn chỉnh.

        Parameters
        ----------
        text_ids : torch.Tensor
            Text token IDs.

            Shape:
                [batch_size, text_length]

        music_input_ids : torch.Tensor
            Music token IDs dùng làm input cho decoder.

            Shape:
                [batch_size, music_length]

        Returns
        -------
        torch.Tensor

            Logits.

            Shape:
                [batch_size, music_length, music_vocab_size]
        """

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        memory, text_padding_mask = self.encode(
            text_ids
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        logits = self.decode(
            music_ids=music_input_ids,
            memory=memory,
            text_padding_mask=text_padding_mask,
        )

        return logits


# ============================================================
# 6. HÀM TẠO MODEL
# ============================================================

def create_model(
    text_vocab_size: int,
    music_vocab_size: int,
    device: torch.device | str | None = None,
) -> TextToMusicTransformer:
    """
    Tạo Text-to-Music Transformer.

    Parameters
    ----------
    text_vocab_size : int
        Số lượng token trong text vocabulary.

    music_vocab_size : int
        Số lượng token trong music vocabulary.

    device : torch.device | str | None
        CPU hoặc CUDA.

    Returns
    -------
    TextToMusicTransformer
        Model đã được đưa lên device.
    """

    model = TextToMusicTransformer(
        text_vocab_size=text_vocab_size,
        music_vocab_size=music_vocab_size,
    )

    if device is not None:
        model = model.to(device)

    return model


# ============================================================
# 7. THÔNG TIN MODEL
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:
    """
    Đếm số lượng trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_summary(
    model: TextToMusicTransformer,
) -> None:
    """
    In thông tin tổng quan của model.
    """

    print("=" * 60)
    print("TEXT-TO-MUSIC TRANSFORMER")
    print("=" * 60)

    print(
        f"Text vocabulary size : "
        f"{model.text_vocab_size}"
    )

    print(
        f"Music vocabulary size: "
        f"{model.music_vocab_size}"
    )

    print(
        f"D_MODEL              : "
        f"{D_MODEL}"
    )

    print(
        f"Attention heads      : "
        f"{N_HEADS}"
    )

    print(
        f"Encoder layers       : "
        f"{NUM_ENCODER_LAYERS}"
    )

    print(
        f"Decoder layers       : "
        f"{NUM_DECODER_LAYERS}"
    )

    print(
        f"Feedforward dimension: "
        f"{DIM_FEEDFORWARD}"
    )

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    print("=" * 60)


# ============================================================
# 8. TEST MODEL
# ============================================================

def test_model() -> None:
    """
    Kiểm tra model có thể forward bình thường hay không.

    Đây chỉ là test kiến trúc.
    Không sử dụng dataset thật.
    """

    print()
    print("=" * 60)
    print("TEST MODEL")
    print("=" * 60)

    # Vocabulary giả lập.
    text_vocab_size = 1000
    music_vocab_size = 500

    # Device.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # Tạo model.
    model = create_model(
        text_vocab_size=text_vocab_size,
        music_vocab_size=music_vocab_size,
        device=device,
    )

    model.eval()

    print_model_summary(model)

    # --------------------------------------------------------
    # Tạo dữ liệu giả lập
    # --------------------------------------------------------

    batch_size = 2
    text_length = 16
    music_length = 32

    text_ids = torch.randint(
        low=1,
        high=text_vocab_size,
        size=(
            batch_size,
            text_length,
        ),
        device=device,
    )

    music_input_ids = torch.randint(
        low=1,
        high=music_vocab_size,
        size=(
            batch_size,
            music_length,
        ),
        device=device,
    )

    # Đặt token đầu tiên là BOS.
    music_input_ids[:, 0] = MUSIC_BOS_ID

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(
            text_ids=text_ids,
            music_input_ids=music_input_ids,
        )

    # --------------------------------------------------------
    # Kiểm tra shape
    # --------------------------------------------------------

    print()
    print("Input:")
    print(
        f"  text_ids        : "
        f"{text_ids.shape}"
    )

    print(
        f"  music_input_ids : "
        f"{music_input_ids.shape}"
    )

    print()
    print("Output:")
    print(
        f"  logits          : "
        f"{logits.shape}"
    )

    expected_shape = (
        batch_size,
        music_length,
        music_vocab_size,
    )

    if tuple(logits.shape) == expected_shape:
        print()
        print("MODEL TEST: PASSED")
    else:
        print()
        print("MODEL TEST: FAILED")

        print(
            f"Expected: {expected_shape}"
        )

        print(
            f"Actual  : {tuple(logits.shape)}"
        )

    print("=" * 60)


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    test_model()
