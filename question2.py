import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

FEATURE_DATA = "features_selected.csv"
PLOT_DIR = "fig/question2"
LUCKY_SEED = 42
PLOT_COLORS = sns.color_palette("Set2")


def load_feature_data(file_path):
    df = pd.read_csv(file_path)
    groups = df["source_index"]
    X = df.drop(columns=["fault_type", "fault_size", "load", "rpm", "source_index"])
    y_str = df["fault_type"]
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    return X, y, groups, le.classes_


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def partition_data_by_group(X, y, groups, class_names, test_size=0.2, seed=LUCKY_SEED):
    y_str = pd.Series(y).map(dict(enumerate(class_names)))
    group_labels = pd.DataFrame({"group": groups, "label": y_str}).drop_duplicates()
    train_groups, test_groups = train_test_split(
        group_labels,
        test_size=test_size,
        random_state=seed,
        stratify=group_labels["label"],
    )
    train_group_names = train_groups["group"].to_list()
    test_group_names = test_groups["group"].to_list()
    train_idx = groups[groups.isin(train_group_names)].index
    test_idx = groups[groups.isin(test_group_names)].index
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_config):
    best_models = {}
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    for name, (model, params) in model_config.items():
        print(f"---------- Training {name} ----------")
        grid_search = GridSearchCV(
            model, params, cv=5, n_jobs=-1, verbose=2, scoring="accuracy"
        )
        if name == "XGBoost":
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(
            f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}"
        )
    return best_models


def evaluate_model(models, X_test, y_test, class_names):
    for name, model in models.items():
        print(f"---------- Evaluating {name} ----------")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"{PLOT_DIR}/confusion_matrix_{name}.pdf")
        plt.close()

        # CM normalized
        cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Normalized Confusion Matrix (Recall) for {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"{PLOT_DIR}/confusion_matrix_normalized_{name}.pdf")
        plt.close()

        # ROC Curve and AUC
        # One class vs rest
        y_prob = model.predict_proba(X_test)
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
            plt.plot(
                fpr, tpr, label=f"{class_name} (AUC = {auc:.2f})", color=PLOT_COLORS[i]
            )
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve for {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"{PLOT_DIR}/roc_curve_{name}.pdf")
        plt.close()


if __name__ == "__main__":
    models = {
        "SVM": (
            SVC(probability=True, random_state=LUCKY_SEED, class_weight="balanced"),
            {
                "C": [1, 10, 100],
                "gamma": ["scale", 0.1, 0.01],
                "kernel": ["rbf"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=LUCKY_SEED, class_weight="balanced"),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="mlogloss", random_state=LUCKY_SEED),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }
    print("Loading feature data...")
    X, y, groups, class_names = load_feature_data(FEATURE_DATA)
    print("Partitioning data...")
    X_train, X_test, y_train, y_test = partition_data_by_group(
        X, y, groups, class_names
    )
    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    print("Training models...")
    best_models = train_model(X_train, y_train, models)
    print("Evaluating models...")
    evaluate_model(best_models, X_test, y_test, class_names)
    print("Saving models and scaler...")
    for name, model in best_models.items():
        joblib.dump(model, f"{name}_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
