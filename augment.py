import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
try:
    import pyloudnorm as pyln
except Exception:
    pyln = None
    
from opus_augment.opus_augment_simulate import OpusAugment

    
def compute_spectrogram_db(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    multiplier: float = 10.0,
    amin: float = 1e-10,
    db_multiplier: float = 0.0
) -> torch.Tensor:
    # 1D → 2D に整形
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, time]
        
    # スペクトログラム計算
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=power
    )
    spec = spec_transform(waveform)  # [1, freq_bins, time_frames]

    # dB 変換
    spec_db = torchaudio.functional.amplitude_to_DB(
        spec,
        multiplier=multiplier,
        amin=amin,
        db_multiplier=db_multiplier
    )

    return spec_db.squeeze(0)  # [freq_bins, time_frames]
            
def rms(wave: torch.Tensor) -> torch.Tensor:
    """Compute root mean square of a waveform."""
    return torch.sqrt(torch.mean(wave.square()))


def adjusted_rms(source_rms: torch.Tensor, snr: float) -> torch.Tensor:
    """Adjust noise RMS for desired SNR in dB."""
    return source_rms / (10 ** (snr / 20))


def mix_signals(source: torch.Tensor, noise: torch.Tensor, snr: float, ref: torch.Tensor=None) -> torch.Tensor:
    """Mix source and noise at the specified SNR."""
    ref = source if ref is None else ref
    ref_rms = rms(ref)
    noise_rms = rms(noise)
    if snr is None or snr >= 60:
        return source
    noise_rms = rms(noise)
    target_noise_rms = adjusted_rms(ref_rms, snr)
    return source + noise * (target_noise_rms / noise_rms)


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with path.open('r') as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate augmented mixtures: source+interference, then ambient noise mix"
    )
    parser.add_argument('--speech-csv', type=Path, required=True,
                        help='CSV with columns [source, speaker, index]')
    parser.add_argument('--interf-csv', type=Path, required=True,
                        help='CSV listing interference files (column: noise)')
    parser.add_argument('--env-csv', type=Path, required=True,
                        help='CSV listing environmental noise files (column: noise)')
    parser.add_argument('--mixed-csv', type=Path, default=None,
                        help='Optional CSV for predefined segments and states')
    parser.add_argument('--output-csv', type=Path, required=True,
                        help='Output CSV for mixture metadata')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--config', type=Path, required=True,
                        help='YAML config for augment parameters')
    parser.add_argument('--bps', type=int, default=0,
                        help='Opus bitrate (bps), 0 for random')
    parser.add_argument('--packet_loss_rate', type=float, default=-1.0,
                        help='Packet loss rate (0-1), <0 for random')
    parser.add_argument('--renew_markov_states', action='store_true',
                        help='Ignore saved states and resimulate')
    return parser.parse_args()


import torch
import torchaudio.functional as F

def bs1770_loudness(waveform: torch.Tensor) -> torch.Tensor:
    """
    ITU-R BS.1770 K-ウェイティングフィルタを通し、
    トラック全体の統合ラウドネス（LUFS）を近似計算する。

    waveform: 1D Tensor [T] (mono, float32, -1.0–1.0)
    戻り値: Tensor(1) 統合ラウドネス [LUFS]
    """
    # 係数は BS.1770-4 に準拠
    # 1) プリフィルタ（高域シェルビング+高域通過）
    b_prefilt = torch.tensor([1.0, -2.0, 1.0], dtype=waveform.dtype)
    a_prefilt = torch.tensor([1.0, -1.99004745483398, 0.99007225036621], dtype=waveform.dtype)

    # 2) シェルビングフィルタ
    b_shelf = torch.tensor([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=waveform.dtype)
    a_shelf = torch.tensor([1.0, -1.69065929318241, 0.73248077421585], dtype=waveform.dtype)

    # フィルタ適用（時系列畳み込み）
    y = F.lfilter(waveform.unsqueeze(0), b_prefilt, a_prefilt).squeeze(0)
    y = F.lfilter(y.unsqueeze(0), b_shelf, a_shelf).squeeze(0)

    # パワーを計算し平均化
    power = y.pow(2).mean()

    # 統合ラウドネスの dBFS 換算
    # 定数 -0.691 は 400 Hz トーンを 60 dB LUFS に合わせる補正項（BS.1770-4 より）
    return -0.691 + 10.0 * torch.log10(power + 1e-12)

def loudness_normalize_bs1770(
    waveform: torch.Tensor,
    sr: int,
    target_lufs: float
) -> torch.Tensor:
    """
    BS.1770 近似ラウドネスで LUFS 正規化を行う。

    waveform: 2D Tensor [1, T] または 1D Tensor [T]
    sr は本実装では使用しません（元サンプリングレートを想定）
    """
    # 1D に
    if waveform.dim() == 2 and waveform.shape[0] == 1:
        x = waveform.squeeze(0)
    else:
        x = waveform

    # K-ウェイティング → 統合ラウドネス取得
    current_lufs = bs1770_loudness(x)

    # ゲイン比
    gain = 10 ** ((target_lufs - current_lufs) / 20.0)

    # 適用
    if waveform.dim() == 2:
        return waveform * gain.unsqueeze(0).unsqueeze(-1)
    else:
        return waveform * gain

def loudness_normalize(waveform: torch.Tensor, sr: int, target_lufs: float) -> torch.Tensor:
    """Normalize waveform to target LUFS per EBU R128."""
    if pyln is not None:
        audio = waveform.squeeze().cpu().numpy().astype(float)
        meter = pyln.Meter(sr)
        current = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, current, target_lufs)
        return torch.from_numpy(normalized).unsqueeze(0).type_as(waveform)
    return loudness_normalize_bs1770(waveform, sr, target_lufs=target_lufs)

def process_utterance(
    row: pd.Series,
    interf_row: pd.Series | None,
    env_row: pd.Series | None,
    opus: OpusAugment,
    cfg: dict,
    args: argparse.Namespace,
    out_dir: Path
) -> dict:
    """Process a single utterance: mix interference, then environmental noise, and opus augment."""
    augment_cfg = cfg['augment']

    # determine use flags and snrs
    use_int = augment_cfg.get('interference', {}).get('use', False)
    snr_int = augment_cfg.get('interference', {}).get('snr', None)
    use_env = augment_cfg.get('environment_noise', {}).get('use', False)
    snr_env = augment_cfg.get('environment_noise', {}).get('snr', None)
    loud_target = augment_cfg.get('loudness_target', None)

    # Paths
    src_path = Path(row['source'])
    interf_path = interf_row['noise'] if interf_row is not None else row['noise']
    env_path = env_row['noise'] if env_row is not None else None

    # Load and normalize source
    source, sr = torchaudio.load(src_path)
    if loud_target is not None:
        source = loudness_normalize(source, sr, loud_target)

    # interference processing
    if use_int:
        interf, _ = torchaudio.load(interf_path)
        if interf.shape[-1] < source.shape[-1]:
          pad_len = source.shape[-1] - interf.shape[-1]
          interf = torch.nn.functional.pad(interf, (0, pad_len))
        else:
          max_start = interf.shape[-1] - source.shape[-1]
          s0 = np.random.randint(0, max_start)
          e0 = s0 + source.shape[-1]
          interf = interf[:, s0:e0]
        mix1 = mix_signals(source, interf, snr_int)
    else:
        mix1 = source
        s0 = e0 = None

    # environmental noise processing
    if use_env:
        env_noise, _ = torchaudio.load(env_path)
        if env_noise.shape[-1] < source.shape[-1]:
          pad_len = source.shape[-1] - env_noise.shape[-1]
          env_seg = torch.nn.functional.pad(env_noise, (0, pad_len))
        else:
          max_start2 = env_noise.shape[-1] - mix1.shape[-1]
          s1 = np.random.randint(0, max_start2)
          e1 = s1 + mix1.shape[-1]
          env_seg = env_noise[:, s1:e1]
        final_mix = mix_signals(mix1, env_seg, snr_env, ref=source)
    else:
        final_mix = mix1
        s1 = e1 = None

    # Opus augment
    prev_states = None
    if use_int and interf_row is not None and not args.renew_markov_states:
        prev_states = np.load(interf_row['markov_states'])
    mixture, bps, plr, resample_sr, states = opus(
        final_mix, args.bps, args.packet_loss_rate, prev_states
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
        'interference': str(interf_path) if use_int else None,
        'env_noise': str(env_path) if use_env else None,
        'start_intf': s0,
        'end_intf': e0,
        'start_env': s1,
        'end_env': e1,
        'length': source.shape[-1],
        'speaker': row['speaker'],
        'index': row['index'],
        'rt60': args.rt60,
        'bps': bps,
        'packet_loss_rate': plr,
        'resample': resample_sr,
        'markov_states': str(state_out)
    }

def simple_mix(source, noise, snr, ref):
    adj_noise = None
    if noise.shape[-1] < source.shape[-1]:
        pad_len = source.shape[-1] - noise.shape[-1]
        adj_noise = torch.nn.functional.pad(noise, (0, pad_len))
    else:
        max_start = noise.shape[-1] - source.shape[-1]
        s0 = np.random.randint(0, max_start)
        e0 = s0 + source.shape[-1]
        adj_noise = noise[:, s0:e0]
    assert adj_noise is not None
    return mix_signals(source,  adj_noise, snr, ref)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    opus = OpusAugment(**cfg['augment']['opus'])

    # Load CSVs
    df_speech = pd.read_csv(args.speech_csv)
    df_interf = pd.read_csv(args.interf_csv)
    df_env    = pd.read_csv(args.env_csv)
    df_mixed  = pd.read_csv(args.mixed_csv) if args.mixed_csv else None

    # Random sampling
    int_sel = df_interf.sample(len(df_speech), replace=True).reset_index(drop=True) if df_mixed is None else None
    env_sel = df_env.sample(len(df_speech), replace=True).reset_index(drop=True)         if df_mixed is None else None

    records = []
    for idx, row in df_speech.iterrows():
        interf_row = int_sel.loc[idx] if int_sel is not None else None
        env_row    = env_sel.loc[idx] if env_sel   is not None else None
        base_row   = df_mixed.loc[idx] if df_mixed is not None else row
        rec = process_utterance(
            base_row, interf_row, env_row, opus,
            cfg, args, out_dir
        )
        records.append(rec)

    pd.DataFrame(records).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
