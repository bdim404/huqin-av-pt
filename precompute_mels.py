import multiprocessing as mp
from pathlib import Path

import numpy as np
import torchaudio

from lbd_train import AUDIO_ROOT, MEL_ROOT, SR, load_audio


def process(wav):
    out = MEL_ROOT / wav.relative_to(AUDIO_ROOT).with_suffix(".npy")
    if out.exists():
        return
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=240, n_mels=128)
    db = torchaudio.transforms.AmplitudeToDB()
    m = db(mel(load_audio(str(wav)))).numpy().astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, m)


if __name__ == "__main__":
    wavs = sorted(AUDIO_ROOT.glob("Erhu-*/**/*.wav"))
    with mp.Pool(6) as pool:
        pool.map(process, wavs, chunksize=16)
    print(f"cached {len(wavs)} mels")
