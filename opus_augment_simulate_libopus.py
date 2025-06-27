###############################################################################
# 依存:  pip install opuslib>=3.0.0 numpy torch torchaudio
###############################################################################
import math, random, numpy as np, torch, torchaudio
from opuslib import Encoder as OpusEncoder
from opuslib import Decoder as OpusDecoder
from opus_augment.packet_loss_simulator import GilbertElliotModel
import ctypes
import opuslib                          # OpusError 用

class OpusAugment(torch.nn.Module):
    """
    Opus エンコード → パケットロス → デコード (FEC/PLC) をシミュレート
    """

    def __init__(
        self,
        sample_rate: int,
        frame_duration: float,
        min_bps: int,
        max_bps: int,
        min_packet_loss_rate: float,
        max_packet_loss_rate: float,
        fec_probability: float = 1.0,        # ロス時に FEC を試す確率 (0.0–1.0)
        loss_behavior: str = "plc",          # "plc" | "zero" | "noise"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration / 1000.0  # ms→s
        self.min_bps, self.max_bps = min_bps, max_bps
        self.min_plr, self.max_plr = min_packet_loss_rate, max_packet_loss_rate
        self.fec_probability = fec_probability
        self.loss_behavior = loss_behavior       # "plc" / "zero" / "noise"
        self.channels, self.bytes_per_sample = 1, 2

    # --------------------------------------------------------------------- #

    def forward(
        self,
        waveform: torch.Tensor,                 # [1, T]  float32
        bps: int = 0,
        plr: float = -1.0,
        received: np.ndarray | None = None,
    ):
        # 1) ビットレート選定 & サンプリングレート決定
        bps = self._select_bitrate(bps)
        target_sr = self._bps_to_rate(bps)
        frame_samples = int(self.frame_duration * target_sr)

        # 2) Resample → PCM16 へ
        resampled = torchaudio.functional.resample(waveform, self.sample_rate, target_sr)
        pcm_bytes = (resampled.squeeze().numpy() * 32767).astype(np.int16).tobytes()

        # 3) Codec 初期化
        enc, dec = self._build_codec(target_sr, bps)
        frame_bytes = frame_samples * self.bytes_per_sample * self.channels

        # 4) 受信可否シーケンス
        num_pkts = math.ceil(len(pcm_bytes) / frame_bytes)
        received = self._simulate_plr(plr, num_pkts, received)

        # 5) フレーム単位で符号化/復号
        decoded = self._process_frames(
            pcm_bytes, enc, dec, received, frame_samples, frame_bytes
        )

        # 6) 元サンプリングレートに戻す
        #dec_tensor = torch.from_numpy(decoded).unsqueeze(0)
        dec_tensor = torch.as_tensor(decoded.copy(), dtype=torch.float32).unsqueeze(0)
        dec_tensor = torchaudio.functional.resample(dec_tensor, target_sr, self.sample_rate)

        # 長さそろえ
        dec_tensor = self._fix_length(dec_tensor, waveform.shape[-1])
        return dec_tensor, bps, plr, target_sr, received

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _select_bitrate(self, bps: int) -> int:
        return random.randint(self.min_bps, self.max_bps) if bps == 0 else bps

    @staticmethod
    def _bps_to_rate(bps: int) -> int:
        return 8000 if bps < 12_000 else 12_000 if bps < 15_000 else 16_000

    def _build_codec(self, sr: int, bps: int):
        #max_bytes = int(bps / (1000 * self.frame_duration * 8))

        enc = OpusEncoder(sr, self.channels, application="audio")
        enc.bitrate = bps
        #enc.set_max_payload_bytes(max_bytes)

        dec = OpusDecoder(sr, self.channels)
        return enc, dec

    def _simulate_plr(self, plr: float, length: int, received: np.ndarray | None):
        if plr < 0:
            plr = random.uniform(self.min_plr, self.max_plr)
        return (GilbertElliotModel(plr=plr).simulate(length)
                if received is None else received)

    # -------------------------------------------- #

    def _process_frames(
            self,
            data: bytes,
            enc: OpusEncoder,
            dec: OpusDecoder,
            received: np.ndarray,
            frame_samples: int,
            frame_bytes: int,
    ) -> np.ndarray:
        decoded_buf = []
        last_frame  = np.zeros(frame_samples, np.float32)

        PcmArray = ctypes.c_int16 * frame_samples      # <class '_ctypes.Array[c_short]'>
        # これを毎フレーム使い回す

        for idx, ok in enumerate(received):
            # ---------- 1) bytes → ctypes.int16[] ----------
            start = idx * frame_bytes
            raw   = data[start : start + frame_bytes].ljust(frame_bytes, b"\x00")
            pcm_ptr = PcmArray.from_buffer_copy(raw)   # 必ず frame_samples 要素
            # ---------- 2) エンコード ----------
            pkt = enc.encode(pcm_ptr, frame_samples)

            # ---------- 3) デコード ----------
            if ok:
                pcm = dec.decode(pkt, frame_samples)
                frame = np.frombuffer(pcm, np.int16).astype(np.float32) / 32767
                last_frame = frame
            else:
                if self.loss_behavior == "plc":
                    # FEC を試行
                    use_fec = random.random() < self.fec_probability
                    try:
                        pcm = dec.decode(pkt, frame_samples, use_fec)
                        frame = np.frombuffer(pcm, np.int16).astype(np.float32) / 32767
                        last_frame = frame
                    except opuslib.OpusError:
                        frame = last_frame            # FEC 失敗 → PLC
                elif self.loss_behavior == "zero":
                    frame = np.zeros(frame_samples, np.float32)
                elif self.loss_behavior == "noise":
                    frame = (torch.randn(frame_samples, dtype=torch.float32)
                             .mul_(0.02)                                   # in-place *
                             .numpy())                    
                else:
                    frame = np.zeros(frame_samples, np.float32)
            decoded_buf.append(frame)
        # ---------- 4) list → 1-D contiguous float32 ----------
        frames  = np.stack(decoded_buf, axis=0).astype(np.float32)   # (N, F)
        decoded = frames.reshape(-1).copy()                          # C 連続バッファ

        return decoded
    
    # Pad / trim
    @staticmethod
    def _fix_length(x: torch.Tensor, length: int):
        return torch.cat([x, torch.zeros(1, length - x.shape[-1])], dim=-1) if x.shape[-1] < length else x[..., :length]
