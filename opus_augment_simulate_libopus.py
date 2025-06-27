###############################################################################
# 依存:  pip install opuslib>=3.0.0 numpy torch torchaudio
###############################################################################
import math, random, numpy as np, torch, torchaudio
from opuslib import Encoder as OpusEncoder
from opuslib import Decoder as OpusDecoder
from opus_augment.packet_loss_simulator import GilbertElliotModel

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
        dec_tensor = torch.from_numpy(decoded).unsqueeze(0)
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

        decoded_buf, last_frame = [], np.zeros(frame_samples, np.float32)

        for idx, ok in enumerate(received):
            start = idx * frame_bytes
            raw = data[start:start + frame_bytes].ljust(frame_bytes, b"\x00")
            pkt = enc.encode(raw, frame_size=frame_samples, max_data_bytes=127)

            if ok:                                    # 正常受信
                pcm = dec.decode(pkt)
                frame = np.frombuffer(pcm, np.int16).astype(np.float32) / 32767
                last_frame = frame
            else:                                     # Lost frame
                if self.loss_behavior == "plc":
                    # FECを試す確率
                    if random.random() < self.fec_probability:
                        try:
                            pcm = dec.decode(pkt, fec=True)  # FEC デコード
                            frame = np.frombuffer(pcm, np.int16).astype(np.float32) / 32767
                            last_frame = frame
                        except opuslib.OpusError:
                            frame = last_frame              # FEC 失敗→PLC
                    else:
                        frame = last_frame                  # PLC
                elif self.loss_behavior == "zero":
                    frame = np.zeros(frame_samples, np.float32)
                elif self.loss_behavior == "noise":
                    frame = np.random.randn(frame_samples).astype(np.float32) * 0.02
                else:
                    frame = np.zeros(frame_samples, np.float32)

            decoded_buf.append(frame)

        return np.concatenate(decoded_buf)

    # Pad / trim
    @staticmethod
    def _fix_length(x: torch.Tensor, length: int):
        return torch.cat([x, torch.zeros(1, length - x.shape[-1])], dim=-1) if x.shape[-1] < length else x[..., :length]
