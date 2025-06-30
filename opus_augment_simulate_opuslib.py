import math, ctypes, random, numpy as np, torch, torchaudio, opuslib
from opus_augment.packet_loss_simulator import GilbertElliotModel

class OpusAugment(torch.nn.Module):
    """
    Opus + in-band FEC + PLC シミュレーション（opuslib 依存）。

    • bps >= 12 000 で FEC ビットを埋め込む
    • ロス直後のフレームを FEC → 復元。失敗時は PLC/Zero/Noise
    """

    def __init__(self,
                 sample_rate: int,
                 frame_duration: float,
                 min_bps: int,
                 max_bps: int,
                 min_packet_loss_rate: float,
                 max_packet_loss_rate: float,
                 fec_probability: float = 1.0,
                 loss_behavior: str = "plc"):

        super().__init__()

        self.sr  = sample_rate
        self.dur = frame_duration / 1000          # sec
        self.min_bps, self.max_bps = min_bps, max_bps
        self.min_plr, self.max_plr = min_packet_loss_rate, max_packet_loss_rate
        self.fec_prob   = fec_probability
        self.loss_mode  = loss_behavior          # "plc"|"zero"|"noise"

        self.channels   = 1
        self.bytes_per_sample = 2                # int16
        self.frame_samp = int(self.sr * self.dur)
        self.frame_byte = self.frame_samp * self.bytes_per_sample

        # ctypes 配列型（エンコーダ呼び出しに必要）
        self.PCM = ctypes.c_int16 * self.frame_samp

    # ------------------------------------------------------------------ #
    def forward(self,
                wav: torch.Tensor,
                bps: int = 0,
                plr: float = -1.0,
                received: np.ndarray | None = None):

        bps = self._pick_bitrate(bps)
        if bps < 12_000:
            raise ValueError("FEC を使うには bps を 12 kbps 以上にしてください")

        tgt_sr     = self._bps_to_rate(bps)
        frame_samp = int(self.dur * tgt_sr)
        frame_byte = frame_samp * self.bytes_per_sample

        # Resample → PCM16 bytes
        resamp = torchaudio.functional.resample(wav, self.sr, tgt_sr)
        pcm16  = (resamp.squeeze().clamp(-1,1)*32767).short().cpu().numpy().tobytes()

        # Opus codec
        enc, dec = self._build_codec(tgt_sr, bps, plr)

        # 受信可否シミュレーション
        n_pkt  = math.ceil(len(pcm16) / frame_byte)
        recv   = self._make_recv(plr, n_pkt, received)

        buf, last, prev_lost = [], np.zeros(frame_samp,np.float32), False

        for i, ok in enumerate(recv):
            chunk = pcm16[i*frame_byte:(i+1)*frame_byte].ljust(frame_byte,b'\0')
            pkt   = enc.encode(self.PCM.from_buffer_copy(chunk), frame_samp)

            if ok:                                     # 正常受信
                if prev_lost:                          # ← 直前ロス → FEC
                    try:
                        pcm_prev = dec.decode(pkt, frame_samp, 1)
                        buf[-1]  = np.frombuffer(pcm_prev,np.int16).astype(np.float32)/32767
                    except opuslib.OpusError:
                        pass
                pcm   = dec.decode(pkt, frame_samp, 0)
                last  = np.frombuffer(pcm, np.int16).astype(np.float32)/32767
                buf.append(last)
                prev_lost = False

            else:                                      # ロス
                if self.loss_mode == "zero":
                    frame = np.zeros(frame_samp, np.float32)
                elif self.loss_mode == "noise":
                    frame = (torch.randn(frame_samp)*0.02).numpy()
                else:                                  # plc / plc+fec
                    frame, prev_lost = last, True
                buf.append(frame)

        out = torch.tensor(np.concatenate(buf), dtype=torch.float32).unsqueeze(0)
        out = torchaudio.functional.resample(out, tgt_sr, self.sr)
        return self._fix(out, wav.shape[-1]), bps, plr, tgt_sr, recv

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _pick_bitrate(self, bps: int) -> int:
        return random.randint(self.min_bps, self.max_bps) if bps == 0 else bps

    def _bps_to_rate(self, bps: int) -> int:
        return 8000 if bps < 12_000 else 12_000 if bps < 15_000 else 16_000

    def _build_codec(self, sr: int, bps: int, plr: float):
        enc = opuslib.Encoder(sr, 1, application="audio")
        enc.bitrate = bps
        enc.set_inband_fec(1)                           # FEC 埋め込む
        enc.set_packet_loss_perc(int(max(plr, self.min_plr)*100))

        dec = opuslib.Decoder(sr, 1)
        return enc, dec

    def _wave_to_int16(self, wav: torch.Tensor):
        return (wav.squeeze().clamp(-1,1)*32767).short().cpu().numpy()

    def _make_recv(self, plr, n, given):
        if plr < 0: plr = random.uniform(self.min_plr, self.max_plr)
        return (GilbertElliotModel(plr=plr).simulate(n) if given is None else given)

    @staticmethod
    def _fix(x: torch.Tensor, length: int):
        return torch.cat([x, torch.zeros(1, length - x.size(-1))], -1) if x.size(-1)<length else x[..., :length]
