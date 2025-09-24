import os
import torch
import pandas as pd
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dann import DANN
from train import BATCH_SIZE, data_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./dann_model.pth"
SOURCE_IMG_DIR = "./spectrograms/source"
TARGET_IMG_DIR = "./spectrograms/target"
RESULT_CSV_PATH = "./target_predictions_dann.csv"
PLOT_DIR = "./fig/question3"
NUM_CLASSES = 4
LUCKY_SEED = 42


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):  # type: ignore
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (*original_tuple, path)


source_dataset = ImageFolderWithPaths(SOURCE_IMG_DIR, transform=data_transform)
target_dataset = ImageFolderWithPaths(TARGET_IMG_DIR, transform=data_transform)

target_loader = DataLoader(dataset=target_dataset, batch_size=BATCH_SIZE, shuffle=False)
source_loader = DataLoader(dataset=source_dataset, batch_size=BATCH_SIZE, shuffle=False)


def predict_and_extract_features(model_path, source_loader, target_loader):
    model = DANN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_paths = []
    source_features = []
    target_features = []
    source_labels = []
    with torch.no_grad():
        for images, _, paths in tqdm(target_loader, desc="Predicting"):
            images = images.to(DEVICE)
            _, class_outputs, _ = model(images, alpha=0)
            _, preds = torch.max(class_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_paths.extend(paths)
        for data, labels, _ in tqdm(source_loader, desc="Extracting Source Features"):
            data = data.to(DEVICE)
            features = model.feature_extractor(data)
            features = features.view(features.size(0), -1)
            source_features.append(features.cpu().numpy())
            source_labels.append(labels.cpu().numpy())

        for data, _, _ in tqdm(target_loader, desc="Extracting Target Features"):
            data = data.to(DEVICE)
            features = model.feature_extractor(data)
            features = features.view(features.size(0), -1)
            target_features.append(features.cpu().numpy())

    source_features = np.concatenate(source_features, axis=0)
    target_features = np.concatenate(target_features, axis=0)
    source_labels = np.concatenate(source_labels, axis=0)

    return all_preds, all_paths, source_features, target_features, source_labels


def aggregate_predictions(preds, paths, class_names):
    orginal_files = [os.path.basename(p).split("_")[0] for p in paths]
    predicted_labels = [class_names[p] for p in preds]
    df_results = pd.DataFrame(
        {"original_file": orginal_files, "predicted_label": predicted_labels}
    )
    final_predictions = (
        df_results.groupby("original_file")["predicted_label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    return final_predictions


def plot_tsne_visualization(
    source_features, target_features, source_labels, class_names
):
    target_labels = np.array(["Target"] * len(target_features))
    source_labels = np.array([class_names[label] for label in source_labels])
    all_features = np.concatenate((source_features, target_features))
    all_labels = np.concatenate((source_labels, target_labels))

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter_without_progress=3000,
        random_state=LUCKY_SEED,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(all_features)
    df_tsne = pd.DataFrame(
        {
            "Dim1": tsne_results[:, 0],
            "Dim2": tsne_results[:, 1],
            "Domain": all_labels,
        }
    )
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x="Dim1",
        y="Dim2",
        hue="Domain",
        legend="full",
        data=df_tsne,
        alpha=0.8,
        s=80,
        palette="Set2",
        edgecolor="white",
    )
    plt.title("t-SNE of the Feature Space (After 2D-CNN + DANN)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title="Domain / Fault Type", fontsize=12)
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/tsne_feature_space_dann.pdf")
    plt.close()


if __name__ == "__main__":
    print("Predicting target domain and extracting features...")
    preds, paths, source_features, target_features, source_labels = (
        predict_and_extract_features(MODEL_PATH, source_loader, target_loader)
    )
    class_names = source_dataset.classes
    final_predictions = aggregate_predictions(preds, paths, class_names)
    final_predictions.to_csv(RESULT_CSV_PATH, index=False)
    print(f"Predictions saved to {RESULT_CSV_PATH}")
    print("Plotting t-SNE visualization...")
    plot_tsne_visualization(
        source_features, target_features, source_labels, class_names
    )
    print(f"t-SNE plot saved to {PLOT_DIR}/tsne_feature_space_dann.pdf")
