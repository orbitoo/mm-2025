import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.io
import seaborn as sns
from scipy.fft import fft
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from skimage.restoration import denoise_wavelet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from tqdm import tqdm

FAULT_TYPE_PAT = r"[/\\](B|IR|OR|N)(.*?)_\d\.mat$"
FAULT_SIZE_PAT = rf"[/\\](\d{4})[/\\]"
LOAD_PAT = r"_(\d)\.mat$"
SENSOR_TYPE = "DE"
WINDOW_SIZE = 8192
OVERLAP = 4096
FAULT_SOURCE_DIR = "./data/source_domain/48kHz_DE_data"
NORMAL_SOURCE_DIR = "./data/source_domain/48kHz_Normal_data"
SAMPLING_RATE = 48000
WAVELET_TYPE = "sym8"
WAVELET_LEVEL = 4
DENOISE_WAVELET_TYPE = "sym8"
BEARING_PARAMS = {
    "n": 9,
    "d": 0.3126,
    "D": 1.537,
}
PLOT_COLORS = sns.color_palette("Set2", 6)
PLOT_DIR = "fig/question1"
PLOT_SIGNAL_LOAD = 1
DISTRIBUTION_PLOT_TYPE = "violin"
SELECT_N_FEATURES = 42


def infer_fault_type(file_path):
    match = re.search(FAULT_TYPE_PAT, file_path)
    if match:
        return match.group(1)
    return "Unknown"


def infer_fault_size(file_path):
    match = re.search(FAULT_SIZE_PAT, file_path)
    if match:
        return int(match.group(1))
    return 0


def infer_load(file_path):
    match = re.search(LOAD_PAT, file_path)
    if match:
        return int(match.group(1))
    return -1


def infer_rpm(mat_data):
    ks = mat_data.keys()
    k = [key for key in ks if "RPM" in key][0]
    return mat_data[k][0, 0]


def read_signal(mat_data):
    k = [key for key in mat_data.keys() if SENSOR_TYPE in key][0]
    signal = mat_data[k].flatten()
    return signal


def read_mat_file(file_path):
    mat_data = scipy.io.loadmat(file_path)
    signal = read_signal(mat_data)
    rpm = infer_rpm(mat_data)
    return signal, rpm


def denoise_signal(signal, wavelet=DENOISE_WAVELET_TYPE):
    return denoise_wavelet(
        signal,
        wavelet=wavelet,
        mode="soft",
        method="BayesShrink",
        rescale_sigma=True,
    )


def sliding_window(signal, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = window_size - overlap
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start : start + window_size])
    return windows


def build_dataframe(data_dir):
    dfs = []

    def process_dir(dir):
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                signal, rpm = read_mat_file(file_path)
                denoised_signal = denoise_signal(signal)
                windows = sliding_window(signal)
                windows_denoised = sliding_window(denoised_signal)
                fault_type = infer_fault_type(file_path)
                fault_size = infer_fault_size(file_path)
                load = infer_load(file_path)
                df = pd.DataFrame(
                    {
                        "original_signal": windows,
                        "signal": windows_denoised,
                        "fault_type": fault_type,
                        "fault_size": fault_size,
                        "load": load,
                        "rpm": rpm,
                    }
                )
                dfs.append(df)

    process_dir(FAULT_SOURCE_DIR)
    process_dir(NORMAL_SOURCE_DIR)
    return pd.concat(dfs, ignore_index=True)


def extract_time_features(signal):
    features = {}
    features["time_mean"] = np.mean(signal)
    features["time_std"] = np.std(signal)
    features["time_rms"] = np.sqrt(np.mean(signal**2))
    features["time_skewness"] = skew(signal)
    features["time_kurtosis"] = kurtosis(signal)
    peak = np.max(np.abs(signal))
    features["time_peak"] = peak
    features["time_peak_to_peak"] = np.ptp(signal)
    features["time_margin_factor"] = peak / (np.mean(np.sqrt(np.abs(signal))) ** 2)
    features["time_shape_factor"] = features["time_rms"] / (np.mean(np.abs(signal)))
    features["time_impulse_factor"] = peak / np.mean(np.abs(signal))
    return features


def extract_freq_features(signal, sampling_rate=SAMPLING_RATE):
    features = {}
    n = len(signal)
    fft_coeffs = np.abs(fft(signal))[: n // 2]  # type: ignore
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    features["freq_center"] = np.sum(freqs * fft_coeffs) / np.sum(fft_coeffs)
    features["freq_rms"] = np.sqrt(np.sum(freqs**2 * fft_coeffs)) / np.sum(fft_coeffs)
    features["freq_std"] = np.sqrt(
        np.sum((freqs - features["freq_center"]) ** 2 * fft_coeffs) / np.sum(fft_coeffs)
    )
    features["freq_kurtosis"] = kurtosis(fft_coeffs)
    features["freq_skewness"] = skew(fft_coeffs)
    return features


def extract_wavelet_features(signal, wavelet=WAVELET_TYPE, level=WAVELET_LEVEL):
    features = {}
    wp = pywt.WaveletPacket(
        data=signal, wavelet=wavelet, mode="symmetric", maxlevel=level
    )
    nodes = wp.get_level(level, order="natural")
    coeffs = [node.data for node in nodes]
    energy = [np.sum(c**2) for c in coeffs]
    total_energy = np.sum(energy)
    features["wavelet_entropy"] = -np.sum(
        (energy / total_energy) * np.log(energy / total_energy + 1e-12)
    )
    for i, e in enumerate(energy):
        features[f"wavelet_energy_{i}"] = e / total_energy
    for i, c in enumerate(coeffs):
        features[f"wavelet_std_{i}"] = np.std(c)
        features[f"wavelet_skewness_{i}"] = skew(c)
        features[f"wavelet_kurtosis_{i}"] = kurtosis(c)
    return features


def extract_fault_freq_features(
    rpm, n=BEARING_PARAMS["n"], d=BEARING_PARAMS["d"], D=BEARING_PARAMS["D"]
):
    f_r = rpm / 60
    bpfo = f_r * n / 2 * (1 - d / D)
    bpfi = f_r * n / 2 * (1 + d / D)
    bsf = f_r * D / d * (1 - (d / D) ** 2)
    ftf = 1 / 2 * f_r * (1 - d / D)
    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def extract_hilbert_features(
    signal, fault_freqs, freq_band=2.0, sampling_rate=SAMPLING_RATE
):
    features = {}
    n = len(signal)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # type: ignore
    envelope_no_mean = envelope - np.mean(envelope)
    envelope_fft_coeffs = np.abs(fft(envelope_no_mean))[: n // 2]  # type: ignore
    features["envelope_kurtosis"] = kurtosis(envelope)
    features["envelope_skewness"] = skew(envelope)
    features["envelope_entropy"] = -np.sum(
        (envelope**2 / np.sum(envelope**2))
        * np.log(envelope**2 / np.sum(envelope**2) + 1e-12)
    )
    features["envelope_rms"] = np.sqrt(np.mean(envelope**2))
    features["envelope_std"] = np.std(envelope)
    features["envelope_peak"] = np.max(envelope)
    fft_freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    peak_freq_index = np.argmax(envelope_fft_coeffs)
    features["envelope_peak_freq"] = fft_freqs[peak_freq_index]
    # for fault_name, fault_hz in fault_freqs.items():
    #     lower_bound = fault_hz - freq_band
    #     upper_bound = fault_hz + freq_band
    #     indices_in_band = np.where(
    #         (fft_freqs >= lower_bound) & (fft_freqs <= upper_bound)
    #     )[0]
    #     amps_in_band = envelope_fft_coeffs[indices_in_band]
    #     max_amp = np.max(amps_in_band) if len(amps_in_band) > 0 else 0
    #     features[f"envelope_{fault_name}_max_amp"] = max_amp
    #     band_energy = np.sum(amps_in_band**2)
    #     features[f"envelope_{fault_name}_band_energy"] = band_energy
    #     band_mean_amp = np.mean(amps_in_band) if len(amps_in_band) > 0 else 0
    #     features[f"envelope_{fault_name}_mean_amp"] = band_mean_amp
    return features


def extract_features(signal, rpm):
    fault_freqs = extract_fault_freq_features(rpm)
    time_feats = extract_time_features(signal)
    freq_feats = extract_freq_features(signal)
    wavelet_feats = extract_wavelet_features(signal)
    envelope_feats = extract_hilbert_features(signal, fault_freqs)
    features = {**time_feats, **freq_feats, **wavelet_feats, **envelope_feats}
    return features


def build_feature_dataframe(df):
    tqdm.pandas(desc="Extracting features")
    feature_df = df.progress_apply(
        lambda row: extract_features(row["signal"], row["rpm"]), axis=1
    ).apply(pd.Series)
    feature_df = df.join(feature_df)
    feature_df = feature_df.drop(columns=["signal", "original_signal"])
    return feature_df


def get_different_fault_signals_index(df, load=PLOT_SIGNAL_LOAD):
    signals = df["signal"]
    normal = signals[(df["fault_type"] == "N") & (df["load"] == load)].index[0]
    ir = signals[(df["fault_type"] == "IR") & (df["load"] == load)].index[0]
    or_ = signals[(df["fault_type"] == "OR") & (df["load"] == load)].index[0]
    b = signals[(df["fault_type"] == "B") & (df["load"] == load)].index[0]
    return {
        "Normal (N)": normal,
        "Inner Race (IR)": ir,
        "Outer Race (OR)": or_,
        "Ball (B)": b,
    }


def plot_original_signals(signals, df, sampling_rate=SAMPLING_RATE):
    fig, axs = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Time-Domain Waveforms of Different Fault Types", fontsize=16)
    for i, (title, index) in enumerate(signals.items()):
        signal = df["original_signal"].iloc[index]
        time = np.arange(len(signal)) / sampling_rate
        axs[i].plot(time, signal)
        axs[i].set_title(title)
        axs[i].set_ylabel("Amplitude")
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(f"{PLOT_DIR}/signal_original.pdf")
    plt.close()


def plot_denoised_signals(signals, df, sampling_rate=SAMPLING_RATE):
    fig, axs = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "Time-Domain Waveforms of Different Fault Types (Denoised)", fontsize=16
    )
    for i, (title, index) in enumerate(signals.items()):
        signal = df["signal"].iloc[index]
        time = np.arange(len(signal)) / sampling_rate
        axs[i].plot(time, signal)
        axs[i].set_title(title)
        axs[i].set_ylabel("Amplitude")
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(f"{PLOT_DIR}/signal_denoised.pdf")
    plt.close()


def plot_signals(signals, df, sampling_rate=SAMPLING_RATE):
    # combine original and denoised plots
    fig, axs = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Time-Domain Waveforms of Different Fault Types", fontsize=16)
    for i, (title, index) in enumerate(signals.items()):
        original_signal = df["original_signal"].iloc[index]
        denoised_signal = df["signal"].iloc[index]
        time = np.arange(len(original_signal)) / sampling_rate
        axs[i].plot(time, original_signal, label="Original", color="blue")
        axs[i].plot(time, denoised_signal, label="Denoised", color="red")
        axs[i].set_title(title)
        axs[i].set_ylabel("Amplitude")
        axs[i].legend()
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(f"{PLOT_DIR}/signal_combined.pdf")
    plt.close()


def plot_noise_signals(signals, df, sampling_rate=SAMPLING_RATE):
    fig, axs = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Time-Domain Waveforms of Different Fault Types (Noise)", fontsize=16)
    for i, (title, index) in enumerate(signals.items()):
        original_signal = df["original_signal"].iloc[index]
        denoised_signal = df["signal"].iloc[index]
        noise_signal = original_signal - denoised_signal
        time = np.arange(len(noise_signal)) / sampling_rate
        axs[i].plot(time, noise_signal)
        axs[i].set_title(title)
        axs[i].set_ylabel("Amplitude")
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(f"{PLOT_DIR}/signal_noise.pdf")
    plt.close()


def plot_outer_race_fault_envelope(
    df, sampling_rate=SAMPLING_RATE, load=PLOT_SIGNAL_LOAD
):
    idx = df[(df["fault_type"] == "OR") & (df["load"] == load)].index[0]
    signal = df["signal"].iloc[idx]
    fault_freqs = extract_fault_freq_features(df["rpm"].iloc[idx])
    bpfo = fault_freqs["bpfo"]
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # type: ignore
    n = len(signal)
    envelope_fft = np.abs(fft(envelope - np.mean(envelope)))[: n // 2]  # type: ignore
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, envelope_fft)
    plt.title("Envelope Spectrum of an Outer Race Fault Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, bpfo * 4)
    for i in range(1, 4):
        plt.axvline(
            bpfo * i,
            color=PLOT_COLORS[i - 1],
            linestyle="--",
            label=f"{i}x BPFO",
        )

    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/envelope_outer_race_fault.pdf")
    plt.close()


def plot_inner_race_fault_envelope(
    df, sampling_rate=SAMPLING_RATE, load=PLOT_SIGNAL_LOAD
):
    idx = df[(df["fault_type"] == "IR") & (df["load"] == load)].index[0]
    signal = df["signal"].iloc[idx]
    fault_freqs = extract_fault_freq_features(df["rpm"].iloc[idx])
    bpfi = fault_freqs["bpfi"]
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # type: ignore
    n = len(signal)
    envelope_fft = np.abs(fft(envelope - np.mean(envelope)))[: n // 2]  # type: ignore
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, envelope_fft)
    plt.title("Envelope Spectrum of an Inner Race Fault Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, bpfi * 4)
    for i in range(1, 4):
        plt.axvline(
            bpfi * i,
            color=PLOT_COLORS[i - 1],
            linestyle="--",
            label=f"{i}x BPFI",
        )

    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/envelope_inner_race_fault.pdf")
    plt.close()


def plot_ball_fault_envelope_bsf(
    df, sampling_rate=SAMPLING_RATE, load=PLOT_SIGNAL_LOAD
):
    idx = df[(df["fault_type"] == "B") & (df["load"] == load)].index[0]
    signal = df["signal"].iloc[idx]
    fault_freqs = extract_fault_freq_features(df["rpm"].iloc[idx])
    bsf = fault_freqs["bsf"]
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # type: ignore
    n = len(signal)
    envelope_fft = np.abs(fft(envelope - np.mean(envelope)))[: n // 2]  # type: ignore
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, envelope_fft)
    plt.title("Envelope Spectrum of a Ball Fault Signal (BSF)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, bsf * 4)
    for i in range(1, 4):
        plt.axvline(
            bsf * i,
            color=PLOT_COLORS[i - 1],
            linestyle="--",
            label=f"{i}x BSF",
        )

    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/envelope_ball_fault_bsf.pdf")
    plt.close()


def plot_ball_fault_envelope_ftf(
    df, sampling_rate=SAMPLING_RATE, load=PLOT_SIGNAL_LOAD
):
    idx = df[(df["fault_type"] == "B") & (df["load"] == load)].index[0]
    signal = df["signal"].iloc[idx]
    fault_freqs = extract_fault_freq_features(df["rpm"].iloc[idx])
    ftf = fault_freqs["ftf"]
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # type: ignore
    n = len(signal)
    envelope_fft = np.abs(fft(envelope - np.mean(envelope)))[: n // 2]  # type: ignore
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[: n // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, envelope_fft)
    plt.title("Envelope Spectrum of a Ball Fault Signal (FTF)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, ftf * 4)
    for i in range(1, 4):
        plt.axvline(
            ftf * i,
            color=PLOT_COLORS[i - 1],
            linestyle="--",
            label=f"{i}x FTF",
        )

    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/envelope_ball_fault_ftf.pdf")
    plt.close()


def plot_distribution(feature_df, plot_feature_map, plot_type=DISTRIBUTION_PLOT_TYPE):
    for feature, (xlabel, ylabel, title) in plot_feature_map.items():
        plt.figure(figsize=(10, 6))
        plot_fun = sns.violinplot if plot_type == "violin" else sns.boxplot
        plot_fun(x="fault_type", y=feature, data=feature_df, palette="Set2")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        name = "violinplot" if plot_type == "violin" else "boxplot"
        plt.savefig(f"{PLOT_DIR}/{name}_{feature}.pdf")
        plt.close()


def plot_pca_visualization(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    pca_df["Fault Type"] = labels.values
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Fault Type",
        palette="Set2",
        data=pca_df,
        legend="full",
        alpha=0.8,
        s=80,
        edgecolor="white",
    )
    plt.title("PCA of the Feature Space", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Fault Type", fontsize=12)
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/pca_feature_space.pdf")
    plt.close()
    explained_variance = pca.explained_variance_ratio_
    print(f"PC1 解释的方差比例: {explained_variance[0]:.2%}")
    print(f"PC2 解释的方差比例: {explained_variance[1]:.2%}")
    print(f"前两个主成分累计解释的方差比例: {np.sum(explained_variance):.2%}")


def plot_tsne_visualization(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter_without_progress=3000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(features_scaled)
    tsne_df = pd.DataFrame(data=tsne_results, columns=["Dim1", "Dim2"])
    tsne_df["Fault Type"] = labels.values
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x="Dim1",
        y="Dim2",
        hue="Fault Type",
        palette="Set2",
        data=tsne_df,
        legend="full",
        alpha=0.8,
        s=80,
        edgecolor="white",
    )
    plt.title("t-SNE of the Feature Space", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title="Fault Type", fontsize=12)
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/tsne_feature_space.pdf")
    plt.close()


def perform_reliefF_feature_selection(features, labels, n_features=SELECT_N_FEATURES):
    X = features.to_numpy()
    y = labels.to_numpy()
    feature_names = features.columns.to_list()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    relief = ReliefF(n_neighbors=10, n_features_to_select=X_scaled.shape[1])
    relief.fit(X_scaled, y)
    scores_df = pd.DataFrame(
        {"Feature": feature_names, "Score": relief.feature_importances_}
    ).sort_values(by="Score", ascending=False)
    print("ReliefF Feature Importance Scores:")
    print(scores_df.head(n_features))
    top_n_scores = scores_df.head(n_features)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Score", y="Feature", data=top_n_scores, palette="viridis")
    plt.title(f"Top {n_features} Most Important Features (ReliefF)", fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.grid(True, axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/reliefF_top_{n_features}_features.pdf")
    plt.close()

    plt.figure(figsize=(12, 8))
    feature_ranks = np.arange(1, len(scores_df) + 1)
    feature_scores = scores_df["Score"].values
    plt.plot(feature_ranks, feature_scores, marker="o", linestyle="-")  # type: ignore
    plt.title("Feature Importance Scores By Rank (Elbow Method)", fontsize=16)
    plt.xlabel("Feature Rank", fontsize=12)
    plt.ylabel("Importance Score", fontsize=12)
    plt.grid(True, linestyle="--")
    for i, n in enumerate([30, 40, 50]):
        plt.axvline(
            x=n, color=PLOT_COLORS[i], linestyle="--", label=f"Potential Cutoff ({n})"
        )
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/reliefF_feature_importance_elbow.pdf")
    plt.close()

    return scores_df


READ_MODE = True

if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    print("Building dataframe...")
    df = build_dataframe(FAULT_SOURCE_DIR)
    print("Extracting features...")
    if READ_MODE:
        feature_df = pd.read_csv("features.csv")
    else:
        feature_df = build_feature_dataframe(df)
        print("Saving to features.csv...")
        feature_df.to_csv("features.csv", index=False)
    print("Plotting signals...")
    signals = get_different_fault_signals_index(df)
    # plot_original_signals(signals, df)
    plot_denoised_signals(signals, df)
    # plot_noise_signals(signals, df)
    # plot_signals(signals, df)
    print("Plotting outer race fault envelope...")
    plot_outer_race_fault_envelope(df)
    print("Plotting inner race fault envelope...")
    plot_inner_race_fault_envelope(df)
    print("Plotting ball fault envelope...")
    plot_ball_fault_envelope_bsf(df)
    plot_ball_fault_envelope_ftf(df)
    features = feature_df.drop(columns=["fault_size", "load", "rpm"]).select_dtypes(
        include=np.number
    )
    labels = feature_df.loc[features.index, "fault_type"]
    print("Plotting PCA visualization...")
    plot_pca_visualization(features, labels)
    print("Plotting t-SNE visualization...")
    plot_tsne_visualization(features, labels)
    print("Performing ReliefF feature selection...")
    perform_reliefF_feature_selection(features, labels)
    feature_sub_df = feature_df[
        ["fault_type", "fault_size", "load", "rpm"] + features.columns.to_list()
    ]
    feature_sub_df.to_csv("features_selected.csv", index=False)
    print("Plotting distributions...")
    plot_feature_map = {
        "wavelet_energy_0": (
            "Fault Type",
            "Normalized Energy",
            "Wavelet Energy (Node 0) by Fault Type",
        ),
        "wavelet_energy_1": (
            "Fault Type",
            "Normalized Energy",
            "Wavelet Energy (Node 1) by Fault Type",
        ),
        "wavelet_energy_7": (
            "Fault Type",
            "Normalized Energy",
            "Wavelet Energy (Node 7) by Fault Type",
        ),
        "freq_rms": (
            "Fault Type",
            "RMS Frequency",
            "Frequency-Domain RMS Frequency by Fault Type",
        ),
        "wavelet_std_1": (
            "Fault Type",
            "Standard Deviation",
            "Wavelet Coefficient Std Dev (Node 1) by Fault Type",
        ),
        "envelope_peak": (
            "Fault Type",
            "Envelope Peak",
            "Envelope Peak Value by Fault Type",
        ),
        "time_peak": (
            "Fault Type",
            "Peak Value",
            "Time-Domain Peak Value by Fault Type",
        ),
        "time_peak_to_peak": (
            "Fault Type",
            "Peak-to-Peak Value",
            "Time-Domain Peak-to-Peak Value by Fault Type",
        ),
        "envelope_entropy": (
            "Fault Type",
            "Entropy",
            "Envelope Entropy by Fault Type",
        ),
        "time_rms": (
            "Fault Type",
            "RMS Value",
            "Time-Domain RMS Value by Fault Type",
        ),
    }
    plot_distribution(feature_df, plot_feature_map)
    plot_distribution(feature_df, plot_feature_map, plot_type="box")
