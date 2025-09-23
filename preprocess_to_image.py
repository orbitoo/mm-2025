import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from question1 import (
    SAMPLING_RATE,
    TARGET_SAMPLING_RATE,
    build_dataframe,
    build_target_dataframe,
    FAULT_SOURCE_DIR,
    NORMAL_SOURCE_DIR,
    TARGET_DIR,
)

N_FFT = 1024
HOP_LENGTH = 256

IMG_SAVE_DIR = "./spectrograms"
SOURCE_IMG_DIR = os.path.join(IMG_SAVE_DIR, "source")
TARGET_IMG_DIR = os.path.join(IMG_SAVE_DIR, "target")

os.makedirs(SOURCE_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(TARGET_IMG_DIR, "unknown"), exist_ok=True)


def save_spectrogram(signal, sampling_rate, save_path):
    stft_result = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    fig = plt.figure(figsize=(4, 4), frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # type: ignore
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(
        D, sr=sampling_rate, hop_length=HOP_LENGTH, x_axis=None, y_axis=None, ax=ax
    )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_source_data(df):
    print("Processing source domain data...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fault_type = row["fault_type"]
        signal = row["signal"]
        class_dir = os.path.join(SOURCE_IMG_DIR, fault_type)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{row['source_index']}_{row['load']}_{row['fault_size']}_{idx}.png"
        save_path = os.path.join(class_dir, filename)
        save_spectrogram(signal, SAMPLING_RATE, save_path)


def process_target_data(df):
    print("Processing target domain data...")
    save_dir = os.path.join(TARGET_IMG_DIR, "unknown")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        signal = row["signal"]
        filename = f"{row['target_index']}_{index}.png"
        save_path = os.path.join(save_dir, filename)
        save_spectrogram(np.array(signal), TARGET_SAMPLING_RATE, save_path)


if __name__ == "__main__":
    print("Building source dataframe...")
    df_source = build_dataframe(FAULT_SOURCE_DIR, NORMAL_SOURCE_DIR)
    print("Building target dataframe...")
    df_target = build_target_dataframe(TARGET_DIR)
    process_source_data(df_source)
    process_target_data(df_target)
    print("Image generation completed.")
