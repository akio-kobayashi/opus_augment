import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn as nn
from typing import Sequence, Tuple

class ReverbAugment(nn.Module):
    """
    Apply room reverberation using the Image Source Method.

    Args:
        sample_rate: Audio sampling rate in Hz.
        rt60_range: Tuple(min_rt60, max_rt60) in seconds.
        snr: Ambient noise SNR for room simulation.
        room_dim: Room dimensions [x, y, z] in meters.
        mic_position: Microphone coordinates [x, y, z] in meters.
        source_position: Nominal source coordinates [x, y, z] in meters.
        loc_jitter: Max jitter [dx, dy, dz] for source position.
    """

    def __init__(
        self,
        sample_rate: int,
        rt60_range: Tuple[float, float],
        snr: float,
        room_dim: Sequence[float],
        mic_position: Sequence[float],
        source_position: Sequence[float],
        loc_jitter: Sequence[float],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_rt60, self.max_rt60 = rt60_range
        self.snr = snr
        self.room_dim = list(room_dim)
        self.mic_position = list(mic_position)
        self.source_position = np.array(source_position)
        self.loc_jitter = np.array(loc_jitter)

        # Default materials for room surfaces
        self.materials = pra.make_materials(
            ceiling="ceiling_plasterboard",
            floor="audience_0.72_m2",
            east="glass_window",
            west="gypsum_board",
            north="gypsum_board",
            south="gypsum_board",
        )

    def get_rt60_range(self) -> Tuple[float, float]:
        """
        Return the (min_rt60, max_rt60) range.
        """
        return self.min_rt60, self.max_rt60

    def forward(
        self,
        waveform: torch.Tensor,
        rt60: float = 0.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply reverberation to the input waveform.

        Args:
            waveform: Tensor of shape (1, T) or (C, T).
            rt60: Desired RT60 in seconds; if <=0, sampled from range.

        Returns:
            (reverberated, used_rt60)
        """
        # Preserve original amplitude
        orig_peak = waveform.abs().max()
        # Convert to 1D numpy
        signal = waveform.squeeze().cpu().numpy().astype(np.float32)
        length = signal.shape[-1]

        # Sample RT60 if not provided
        if rt60 <= 0:
            rt60 = np.random.uniform(self.min_rt60, self.max_rt60)

        # Compute required image source order
        _, max_order = pra.inverse_sabine(rt60, self.room_dim)

        # Create room and place mic and source
        room = pra.ShoeBox(
            self.room_dim,
            fs=self.sample_rate,
            materials=self.materials,
            max_order=max_order
        )
        room.add_microphone(self.mic_position)
        jitter = (np.random.rand(3) * 2 - 1) * self.loc_jitter
        room.add_source((self.source_position + jitter).tolist(), signal=signal)

        # Simulate with SNR
        room.simulate(self.snr)
        rvb = room.mic_array.signals[0, :length]

        # Convert back to tensor and restore peak
        rvb_tensor = torch.from_numpy(rvb).unsqueeze(0)
        if orig_peak > 0:
            rvb_tensor = rvb_tensor / rvb_tensor.abs().max() * orig_peak

        return rvb_tensor.type_as(waveform), rt60
