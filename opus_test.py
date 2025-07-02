#!/usr/bin/env python3
import argparse
import time
import random
import torchaudio
import torch
import numpy as np
import ctypes
from ctypes import c_int16
from opuslib import Encoder, Decoder
from packet_loss_simulator import GilbertElliotModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Opus encode/decode with FEC/PLC, GE-model loss, and network delay simulation"
    )
    parser.add_argument("input_wav", help="入力 WAV ファイルパス")
    parser.add_argument("output_wav", help="出力（デコード後） WAV ファイルパス")
    parser.add_argument("--fec", action="store_true", help="in-band FEC を有効化")
    parser.add_argument("--plc", action="store_true", help="PLC を有効化")
    parser.add_argument("--bitrate", type=int, default=64000, help="Opus ビットレート (bps)")
    parser.add_argument("--frame_ms", type=int, default=20, help="フレーム長 (ms)")
    parser.add_argument("--plr", type=float, default=0.1, help="GE モデルの長期損失率")
    parser.add_argument("--lamda", type=float, default=0.5, help="GE モデルのバーストパラメータ λ")
    parser.add_argument("--p_good", type=float, default=0.0, help="Good 状態での損失率")
    parser.add_argument("--p_bad", type=float, default=0.5, help="Bad 状態での損失率")
    parser.add_argument("--base_delay_ms", type=float, default=50.0,
                        help="固定遅延 (ms)")
    parser.add_argument("--jitter_std_ms", type=float, default=5.0,
                        help="ジッタ標準偏差 (ms)")
    parser.add_argument("--buffer_ms", type=float, default=60.0,
                        help="受信バッファ長 (ms)")
    return parser.parse_args()

def main():
    args = parse_args()
    # --- WAV 読み込み・前処理 ---
    wav, sr = torchaudio.load(args.input_wav)
    wav = wav.mean(dim=0).numpy()
    frame_size = int(sr * args.frame_ms / 1000)
    num_frames = int(np.ceil(len(wav) / frame_size))
    pad = num_frames * frame_size - len(wav)
    wav = np.pad(wav, (0, pad), 'constant')

    # --- Opus 初期化 ---
    encoder = Encoder(sr, 1, 'audio')
    decoder = Decoder(sr, 1)
    encoder.encoder_state  # internal state
    # FEC 設定
    from opuslib.api.encoder import encoder_ctl
    import opuslib.api.ctl as ctl
    encoder_ctl(encoder.encoder_state, ctl.set_inband_fec, 1 if args.fec else 0)
    encoder_ctl(encoder.encoder_state, ctl.set_packet_loss_perc, int(args.plr*100))

    # --- 損失モデル生成 ---
    model = GilbertElliotModel(plr=args.plr, lamda=args.lamda, p_good=args.p_good, p_bad=args.p_bad)
    recv_seq = model.simulate(num_frames)

    # --- エンコード & ネットワークスケジューリング ---
    events = []
    now = time.time()
    interval = args.frame_ms/1000.0
    for i in range(num_frames):
        # PCM → int16 array
        start = i * frame_size
        pcm_int16 = (wav[start:start+frame_size]*32767).astype(np.int16)
        pcm_buf = (c_int16 * frame_size)(*pcm_int16.tolist())
        packet  = encoder.encode(pcm_buf, frame_size)        

        # 到着時刻＆バッファ
        net_delay     = args.base_delay_ms + random.gauss(0, args.jitter_std_ms)
        arrival       = now + i*interval + net_delay/1000.0
        decode_time   = arrival + args.buffer_ms/1000.0
        pkt_or_none   = packet if recv_seq[i]==1 else None
        events.append((i, decode_time, pkt_or_none))

    # --- デコード & 出力組み立て ---
    decoded = np.zeros(num_frames*frame_size, dtype=np.int16)
    events.sort(key=lambda x: x[1])
    for idx, decode_time, pkt in events:
        while time.time() < decode_time:
            time.sleep(0.001)

        if pkt is None:
            if args.plc:
                # None の代わりに必ず空バイト列 b'' を渡す
                pcm_bytes = decoder.decode(b'', frame_size, decode_fec=0)
            else:
                pcm_bytes = (np.zeros(frame_size, dtype=np.int16)).tobytes()
        else:
            pcm_bytes = decoder.decode(pkt, frame_size, decode_fec=1 if args.fec else 0)

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        decoded[idx*frame_size:(idx+1)*frame_size] = pcm

    # --- WAV 保存 ---
    out = torch.from_numpy(decoded.astype(np.float32)/32767.0).unsqueeze(0)
    torchaudio.save(args.output_wav, out, sr)
    print("Done.")

if __name__=="__main__":
    main()
    
