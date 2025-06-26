import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml

from opus_augment.opus_augment_simulate import OpusAugment
from opus_augment.reverb_augment import ReverbAugment

def rms(wave: torch.Tensor) -> torch.Tensor:
    """Compute root mean square of a waveform."""
    return torch.sqrt(torch.mean(wave.square()))


def adjusted_rms(source_rms: torch.Tensor, snr: float) -> torch.Tensor:
    """Adjust noise RMS for desired SNR in dB."""
    return source_rms / (10 ** (snr / 20))


def mix_signals(source: torch.Tensor, noise: torch.Tensor, snr: float) -> torch.Tensor:
    """Mix source and noise at the specified SNR."""
    if snr >= 60:
        return source
    src_rms = rms(source)
    noise_rms = rms(noise)
    target_noise_rms = adjusted_rms(src_rms, snr)
    return source + noise * (target_noise_rms / noise_rms)


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with path.open('r') as f:
        return yaml.safe_load(f)


def init_reverb_funcs(reverb_cfg: dict) -> tuple[ReverbAugment, ReverbAugment]:
    """Initialize reverb augment functions for source and noise."""
    params = reverb_cfg['params']
    # Fix location ranges to zero vector
    zero_range = [0, 0, 0]
    src_cfg = {**params, 'source_loc': reverb_cfg['source_loc'], 'loc_range': zero_range}
    noise_cfg = {**params, 'source_loc': reverb_cfg['noise_loc'], 'loc_range': zero_range}

    src_func = ReverbAugment(**src_cfg)
    noise_func = ReverbAugment(**noise_cfg)
    return src_func, noise_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate opus augmentations: band-limited, reverb, noise mix, packet loss."
    )
    parser.add_argument('--speech-csv', type=Path, required=True)
    parser.add_argument('--noise-csv', type=Path, required=True)
    parser.add_argument('--mixed-csv', type=Path)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--snr', type=float, default=20)
    parser.add_argument('--rt60', type=float, default=0.2)
    parser.add_argument('--bps', type=int, default=16_000)
    parser.add_argument('--packet-loss', type=float, default=0.1)
    parser.add_argument('--renew-markov', action='store_true')
    return parser.parse_args()


def process_utterance(
    row: pd.Series,
    noise_row: pd.Series | None,
    reverb_src: ReverbAugment,
    reverb_noise: ReverbAugment,
    opus: OpusAugment,
    args: argparse.Namespace,
    out_dir: Path
) -> dict:
    """Process one utterance: load, reverb, mix, opus augment, save."""
    src_path = Path(row['source'])
    noise_path = Path(noise_row['noise']) if noise_row is not None else Path(row['noise'])

    # Load audio
    source, sr = torchaudio.load(src_path)
    noise, _ = torchaudio.load(noise_path)

    # Determine noise segment
    if noise_row is None:
        max_start = noise.shape[-1] - source.shape[-1]
        start = np.random.randint(0, max_start)
        end = start + source.shape[-1]
    else:
        start, end = int(row['start']), int(row['end'])
    noise = noise[:, start:end]

    # Apply reverb if needed
    if args.rt60 > 0.2:
        source = reverb_src(source, args.rt60)[0]
        noise = reverb_noise(noise, args.rt60)[0]

    # Mix source and noise
    mixture = mix_signals(source, noise, args.snr)

    # Opus augment
    markov_states = None
    if row.get('markov_states') and not args.renew_markov:
        markov_states = np.load(row['markov_states'])
    mixture, bps, packet_loss, resample, states = opus(
        mixture, args.bps, args.packet_loss, markov_states
    )

    # Save outputs
    basename = src_path.stem
    mix_out = out_dir / f"{basename}_mix.wav"
    state_out = out_dir / f"{basename}.npy"
    np.save(state_out, np.array(states))
    torchaudio.save(mix_out, mixture.cpu(), sr)

    return {
        'mixture': str(mix_out),
        'source': str(src_path),
        'noise': str(noise_path),
        'start': start,
        'end': end,
        'length': source.shape[-1],
        'speaker': row['speaker'],
        'index': row['index'],
        'snr': args.snr,
        'rt60': args.rt60,
        'bps': bps,
        'packet_loss_rate': packet_loss,
        'resample': resample,
        'markov_states': str(state_out)
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reverb_src, reverb_noise = init_reverb_funcs(cfg['augment']['reverb'])
    opus = OpusAugment(**cfg['augment']['opus'])

    df_speech = pd.read_csv(args.speech_csv)
    df_noise = pd.read_csv(args.noise_csv)
    df_mixed = pd.read_csv(args.mixed_csv) if args.mixed_csv else None

    noise_sel = None
    if df_mixed is None:
        noise_sel = df_noise.sample(len(df_speech), replace=True).reset_index(drop=True)

    records = []
    for idx, speech_row in df_speech.iterrows():
        noise_row = None if df_mixed is None else None
        if noise_sel is not None:
            noise_row = noise_sel.loc[idx]
        row = df_mixed.loc[idx] if df_mixed is not None else speech_row
        record = process_utterance(
            row, noise_row, reverb_src, reverb_noise, opus, args, out_dir
        )
        records.append(record)

    pd.DataFrame(records).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
