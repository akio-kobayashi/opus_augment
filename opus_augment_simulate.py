import math
from pathlib import Path
import random
import numpy as np
import torch
import torchaudio
from pyogg import OpusEncoder, OpusDecoder
from opus_augment.packet_loss_simulator import GilbertElliotModel

class OpusAugment(torch.nn.Module):
    """
    Apply Opus encoding/decoding with simulated packet loss.
    Args:
        sample_rate: Original sample rate (Hz).
        frame_duration: Frame length in ms.
        min_bps, max_b1ps: Bitrate range (bps).
        min_packet_loss_rate, max_packet_loss_rate: PLR range.
        fec_probability: FEC/PLC decode rate for lost packets.
    """
    def __init__(
        self,
        sample_rate: int,
        frame_duration: float,
        min_bps: int,
        max_bps: int,
        min_packet_loss_rate: float,
        max_packet_loss_rate: float,
        fec_probability: float = 1.0,
        loss_behavior: str = "plc",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration / 1000  # ms -> s
        self.min_bps = min_bps
        self.max_bps = max_bps
        self.min_plr = min_packet_loss_rate
        self.max_plr = max_packet_loss_rate
        self.fec_probability = fec_probability
        self.channels = 1
        self.bytes_per_sample = 2
        self.loss_behavior = loss_behavior  # "plc", "zero", or "noise"

    def forward(
        self,
        waveform: torch.Tensor,
        bps: int = 0,
        plr: float = -1.0,
        received: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, int, float, int, np.ndarray]:
        # Select bitrate and compute target rate
        bps = self._select_bitrate(bps)
        target_sr = self._bps_to_rate(bps)
        frame_samples = int(self.frame_duration * target_sr)

        # Resample to Opus rate
        resampled = torchaudio.functional.resample(
            waveform, self.sample_rate, target_sr
        )

        # Initialize codec
        encoder, decoder = self._build_codec(target_sr, bps)

        # Convert to PCM bytes
        pcm_bytes = self._waveform_to_bytes(resampled)

        # Simulate packet reception
        num_pkts = math.ceil(len(pcm_bytes) / (frame_samples * self.bytes_per_sample))
        received = self._simulate_plr(plr, num_pkts, received)

        # Encode/decode per frame
        decoded = self._process_frames(
            pcm_bytes, encoder, decoder, received, frame_samples
        )

        # Convert back to tensor and original rate
        decoded_tensor = torch.from_numpy(decoded).unsqueeze(0)
        decoded_tensor = torchaudio.functional.resample(
            decoded_tensor, target_sr, self.sample_rate
        )

        # Trim or pad to original length
        decoded_tensor = self._fix_length(decoded_tensor, waveform.shape[-1])

        return decoded_tensor, bps, plr, target_sr, received

    def _select_bitrate(self, bps: int) -> int:
        if bps == 0:
            return random.randint(self.min_bps, self.max_bps)
        return bps

    def _bps_to_rate(self, bps: int) -> int:
        if bps < 12000:
            return 8000
        if bps < 15000:
            return 12000
        return 16000

    def _build_codec(self, sr: int, bps: int) -> tuple[OpusEncoder, OpusDecoder]:
        # Frame size per packet (bytes)
        max_bytes = int(bps / (1000 * self.frame_duration * 8))

        enc = OpusEncoder()
        enc.set_application("voip")
        enc.set_sampling_frequency(sr)
        enc.set_channels(self.channels)
        enc.set_max_bytes_per_frame(max_bytes)

        dec = OpusDecoder()
        dec.set_channels(self.channels)
        dec.set_sampling_frequency(sr)

        return enc, dec

    def _waveform_to_bytes(self, wav: torch.Tensor) -> bytes:
        arr = (wav.cpu().numpy().squeeze() * 32767).astype(np.int16)
        return arr.tobytes()

    def _simulate_plr(
        self,
        plr: float,
        num_pkts: int,
        received: np.ndarray | None,
    ) -> np.ndarray:
        if plr < 0:
            plr = random.uniform(self.min_plr, self.max_plr)            
        if received is None:
            model = GilbertElliotModel(plr=plr)
            received = model.simulate(num_pkts)
        return received

    def _process_frames(
        self,
        data: bytes,
        enc: OpusEncoder,
        dec: OpusDecoder,
        received: np.ndarray,
        frame_samples: int,
    ) -> np.ndarray:
        frame_bytes = frame_samples * self.bytes_per_sample * self.channels
        decoded_buf = []
        # 最後に復号できたフレームを保存（PLC用）
        last_frame: np.ndarray = np.zeros((frame_samples,), dtype=np.float32)

        for idx, recv in enumerate(received):
            start = idx * frame_bytes
            chunk = data[start:start + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))

            pkt = enc.encode(chunk)

            if recv:
                out = dec.decode(pkt)
                frame = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767
                last_frame = frame
            else:
                if self.loss_behavior == "plc":
                    if random.random() < self.fec_probability:
                        out = dec.decode(pkt, decode_fec=True)
                        frame = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767
                        last_frame = frame
                    else:
                        frame = last_frame
                elif self.loss_behavior == "zero":
                    frame = np.zeros((frame_samples,), dtype=np.float32)
                elif self.loss_behavior == "noise":
                    # ホワイトノイズ
                    noise = (np.random.randn(frame_samples) * 0.02).astype(np.float32)
                    decoded_buf.append(noise)
                    continue
                else:
                    # それ以外はゼロで埋め
                    frame = np.zeros((frame_samples,), dtype=np.float32)
            decoded_buf.append(frame)
            #decoded_buf.append(
            #    np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767
            #)

        return np.concatenate(decoded_buf)

    def _fix_length(self, wav: torch.Tensor, length: int) -> torch.Tensor:
        if wav.shape[-1] < length:
            pad = torch.zeros(1, length - wav.shape[-1])
            return torch.cat([wav, pad], dim=-1)
        return wav[:, :length]
