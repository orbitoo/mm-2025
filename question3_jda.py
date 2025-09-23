import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from jda import JDA
from scipy.stats import entropy

SOURCE_FILE = "features.csv"
TARGET_FILE = "target_features.csv"
JDA_PARAMS = {
    "k": 40,
    "lambda_": 0.08,
    "ker": "rbf",
    "gamma": 1e-5,
    "T": 20,
}
PLOT_DIR = "fig/question3"
LUCKY_SEED = 42
CONFIDENCE_THRESHOLD = 0.9


def load_data(source_file, target_file):
    df_src = pd.read_csv(source_file)
    df_tgt = pd.read_csv(target_file)

    feature_cols = [
        c
        for c in df_src.columns
        if c not in ["fault_type", "fault_size", "load", "rpm", "source_index"]
    ]
    Xs = df_src[feature_cols].values
    Xt = df_tgt[feature_cols].values

    le = LabelEncoder()
    ys_str = df_src["fault_type"]
    ys = le.fit_transform(ys_str)

    scaler = StandardScaler()
    Xs_scaled = scaler.fit_transform(Xs)
    Xt_scaled = scaler.transform(Xt)

    return Xs_scaled, ys, Xt_scaled, ys_str, le.classes_


def run_jda(
    Xs,
    Ys,
    Xt,
    k=JDA_PARAMS["k"],
    lambda_=JDA_PARAMS["lambda_"],
    ker=JDA_PARAMS["ker"],
    gamma=JDA_PARAMS["gamma"],
    T=JDA_PARAMS["T"],
):
    jda_transformer = JDA(k=k, lambda_=lambda_, ker=ker, gamma=gamma, T=T)  # type: ignore
    Xs_new, Xt_new = jda_transformer.fit_predict(Xs, Ys, Xt)
    return Xs_new, Xt_new


def final_classifier(Xs_new, Ys, Xt_new):
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=LUCKY_SEED,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
    )
    clf.fit(Xs_new, Ys)
    Yt_pred = clf.predict(Xt_new)
    return Yt_pred, clf


def plot_tsne(Xs_new, Xt_new, ys_str, name="tsne_feature_space_jda"):
    combined_features = np.vstack((Xs_new, Xt_new))
    source_labels = ys_str.values
    target_labels = np.array(["Target"] * len(Xt_new))
    combined_labels = np.concatenate((source_labels, target_labels))
    tsne = TSNE(
        n_components=2,
        random_state=LUCKY_SEED,
        perplexity=50,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(combined_features)
    df_tsne = pd.DataFrame(tsne_results, columns=["Dim1", "Dim2"])
    df_tsne["Label"] = combined_labels
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x="Dim1",
        y="Dim2",
        hue="Label",
        legend="full",
        palette="Set2",
        data=df_tsne,
        s=80,
        alpha=0.8,
        edgecolor="white",
    )
    plt.title("t-SNE of the Feature Space (After JDA)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title="Domain / Fault Type", fontsize=12)
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/{name}.pdf")
    plt.close()


def fine_tuning(k_list, lambda_list, gamma_list, Xs, Ys, Xt):
    best_score = 0
    best_params = {}
    best_predictions = None
    for k in k_list:
        for lambda_ in lambda_list:
            for gamma in gamma_list:
                print(f"----------k={k}, lambda={lambda_}, gamma={gamma}----------")
                Xs_new, Xt_new = run_jda(Xs, Ys, Xt, k=k, lambda_=lambda_, gamma=gamma)
                Yt_pred, clf = final_classifier(Xs_new, Ys, Xt_new)
                target_prob = clf.predict_proba(Xt_new)  # type: ignore
                max_prob = np.max(target_prob, axis=1)
                avg_prob = np.mean(max_prob)

                _, counts = np.unique(Yt_pred, return_counts=True)
                pred_label_dist = counts / counts.sum()
                pred_entropy = entropy(pred_label_dist, base=2)
                print(f"Average max probability: {avg_prob:.4f}")
                print(f"Prediction entropy: {pred_entropy:.4f}")
                score = avg_prob * pred_entropy
                print(f"Combined score (Confidence * Entropy): {score:.4f}")

                if score > best_score:
                    print(f"!!! New best score: {score:.4f} !!!")
                    best_score = score
                    best_params = {"k": k, "lambda_": lambda_, "gamma": gamma}
                    best_predictions = Yt_pred
                    plot_tsne(
                        Xs_new,
                        Xt_new,
                        Ys_str,
                        name=f"tsne_feature_space_jda_k{k}_lambda{lambda_}_gamma{gamma}",
                    )

    print("---------- Best Parameters ----------")
    print(best_params)
    return best_predictions, best_params


def self_training_refinement(
    Xs_new, Ys, Xt_new, confidence_threshold=CONFIDENCE_THRESHOLD
):
    print("---------- Self-Training Refinement ----------")
    initial_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=LUCKY_SEED,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
    )
    initial_clf.fit(Xs_new, Ys)
    target_prob = initial_clf.predict_proba(Xt_new)  # type: ignore
    max_prob = np.max(target_prob, axis=1)
    high_confidence_indices = np.where(max_prob >= confidence_threshold)[0]
    final_predictions = np.zeros(len(Xt_new), dtype=int)
    if len(high_confidence_indices) > 5:
        print(
            f"Adding {len(high_confidence_indices)} high-confidence target samples to source."
        )
        Xt_high_conf = Xt_new[high_confidence_indices]
        pseudo_labels = initial_clf.predict(Xt_high_conf)
        final_predictions[high_confidence_indices] = pseudo_labels
        X_train_augmented = np.vstack((Xs_new, Xt_high_conf))
        y_train_augmented = np.concatenate((Ys, pseudo_labels))
        print("Re-training classifier on augmented dataset...")
        final_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=LUCKY_SEED,
            max_depth=10,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features="sqrt",
            class_weight="balanced",
        )
        final_clf.fit(X_train_augmented, y_train_augmented)
        low_confidence_indices = np.where(max_prob < confidence_threshold)[0]
        if len(low_confidence_indices) > 0:
            print(
                f"Predicting labels for {len(low_confidence_indices)} low-confidence samples."
            )
            low_confidence_samples = Xt_new[low_confidence_indices]
            low_confidence_preds = final_clf.predict(low_confidence_samples)
            final_predictions[low_confidence_indices] = low_confidence_preds
    else:
        print("Not enough high-confidence samples found. Skipping self-training.")
        final_clf = initial_clf
        final_predictions = initial_clf.predict(Xt_new)
    return final_predictions, final_clf


TUNE_MODE = False
SELF_TRAINING_MODE = True

if __name__ == "__main__":
    print("Loading data...")
    Xs, Ys, Xt, Ys_str, class_names = load_data(SOURCE_FILE, TARGET_FILE)
    print("Running JDA...")
    if TUNE_MODE:
        k_list = [40]
        lambda_list = [0.08, 0.1, 0.2]
        gamma_list = [1e-5]
        Yt_pred, best_params = fine_tuning(k_list, lambda_list, gamma_list, Xs, Ys, Xt)
    else:
        Xs_new, Xt_new = run_jda(Xs, Ys, Xt)
        if SELF_TRAINING_MODE:
            Yt_pred, final_classifier_model = self_training_refinement(
                Xs_new, Ys, Xt_new
            )
        else:
            Yt_pred, _ = final_classifier(Xs_new, Ys, Xt_new)
        plot_tsne(Xs_new, Xt_new, Ys_str)
    Yt_pred_str = class_names[Yt_pred]
    df_target_info = pd.read_csv(TARGET_FILE)[["target_index"]]
    df_target_info["predicted_label"] = Yt_pred_str
    file_labels = df_target_info.groupby("target_index")["predicted_label"].agg(
        lambda x: x.mode()[0]
    )
    print("---------- Final Predicted Labels ----------")
    print(file_labels)
    file_labels.to_csv("target_predictions_jda.csv")
