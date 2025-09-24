import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

from dann import DANN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOURCE_IMG_DIR = "./spectrograms/source"
TARGET_IMG_DIR = "./spectrograms/target"
MODEL_SAVE_PATH = "./dann_model.pth"

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
K_STEPS_D = 1
ALPHA = 0.2
BETA = 0.6
CLIP_VALUE = 1.0
LEARNING_RATE_MULTIPLIER_D = 0.5
LUCKY_SEED = 42
PLOT_COLORS = sns.color_palette("Set2", 4)
PLOT_DIR = "./fig/question3"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    XX = kernels[:batch_size, :batch_size]  # type: ignore
    YY = kernels[batch_size:, batch_size:]  # type: ignore
    XY = kernels[:batch_size, batch_size:]  # type: ignore
    YX = kernels[batch_size:, :batch_size]  # type: ignore
    loss = torch.mean(XX + YY - XY - YX)
    return loss


data_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

source_dataset = datasets.ImageFolder(SOURCE_IMG_DIR, transform=data_transform)
target_dataset = datasets.ImageFolder(TARGET_IMG_DIR, transform=data_transform)

source_loader = DataLoader(
    dataset=source_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

target_loader = DataLoader(
    dataset=target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

if __name__ == "__main__":
    set_seed(LUCKY_SEED)
    print(f"Using device: {DEVICE}")
    print(
        f"Source dataset size: {len(source_dataset)}, classes: {source_dataset.classes}"
    )
    print(f"Target dataset size: {len(target_dataset)}")
    class_counts = np.bincount(source_dataset.targets)
    total_samples = len(source_dataset.targets)
    num_classes = len(source_dataset.classes)
    class_weights = total_samples / (num_classes * class_counts)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    model = DANN(num_classes=len(source_dataset.classes)).to(DEVICE)
    optimizer_F = optim.Adam(
        list(model.feature_extractor.parameters())
        + list(model.label_predictor.parameters()),
        lr=LEARNING_RATE,
    )
    optimizer_D = optim.Adam(
        model.domain_discriminator.parameters(),
        lr=LEARNING_RATE * LEARNING_RATE_MULTIPLIER_D,
    )

    scheduler_F = optim.lr_scheduler.StepLR(
        optimizer=optimizer_F, step_size=20, gamma=0.1
    )
    scheduler_D = optim.lr_scheduler.StepLR(
        optimizer=optimizer_D, step_size=20, gamma=0.1
    )

    loss_class = nn.CrossEntropyLoss(weight=class_weights_tensor)
    loss_domain = nn.CrossEntropyLoss()

    history = {
        "loss_class": [],
        "loss_domain_d": [],
        "loss_adversarial_f": [],
        "loss_mmd": [],
    }

    print("Starting alternating training with DANN + MMD...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        target_iter = iter(target_loader)

        running_loss_class = 0.0
        running_loss_domain_d = 0.0
        running_loss_adversarial_f = 0.0
        running_loss_mmd = 0.0

        len_dataloader = len(source_loader)

        for i, (source_data, source_label) in enumerate(
            tqdm(source_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        ):
            source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
            domain_label_source = torch.zeros(source_data.size(0)).long().to(DEVICE)

            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
            domain_label_source = torch.zeros(source_data.size(0)).long().to(DEVICE)

            p = float(i + epoch * len_dataloader) / (NUM_EPOCHS * len_dataloader)
            alpha = ALPHA
            for param in model.domain_discriminator.parameters():
                param.requires_grad = True

            for _ in range(K_STEPS_D):
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                target_data = target_data.to(DEVICE)
                domain_label_target = torch.ones(target_data.size(0)).long().to(DEVICE)

                optimizer_D.zero_grad()

                _, _, domain_output_source = model(source_data)
                _, _, domain_output_target = model(target_data)

                err_s_domain = loss_domain(domain_output_source, domain_label_source)
                err_t_domain = loss_domain(domain_output_target, domain_label_target)
                err_domain = err_s_domain + err_t_domain

                err_domain.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.domain_discriminator.parameters(), CLIP_VALUE
                )
                optimizer_D.step()

            for param in model.domain_discriminator.parameters():
                param.requires_grad = False

            optimizer_F.zero_grad()

            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            target_data = target_data.to(DEVICE)
            source_features, class_output, domain_output_source = model(source_data)
            target_features, _, _ = model(target_data)

            err_s_label = loss_class(class_output, source_label)
            err_adversarial = loss_domain(
                domain_output_source, torch.ones_like(domain_label_source)
            )
            err_mmd = mmd_loss(source_features, target_features)
            total_loss_F = err_s_label + alpha * err_adversarial + BETA * err_mmd
            total_loss_F.backward()
            torch.nn.utils.clip_grad_norm_(
                model.feature_extractor.parameters(), CLIP_VALUE
            )
            torch.nn.utils.clip_grad_norm_(
                model.label_predictor.parameters(), CLIP_VALUE
            )

            optimizer_F.step()

            running_loss_class += err_s_label.item()
            running_loss_domain_d += err_domain.item()  # type: ignore
            running_loss_adversarial_f += err_adversarial.item()
            running_loss_mmd += err_mmd.item()

        avg_loss_class = running_loss_class / len_dataloader
        avg_loss_domain_d = running_loss_domain_d / len_dataloader
        avg_loss_adversarial_f = running_loss_adversarial_f / len_dataloader
        avg_loss_mmd = running_loss_mmd / len_dataloader

        scheduler_F.step()
        scheduler_D.step()

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
            f"Class Loss: {avg_loss_class:.4f} | "
            f"Domain Loss (D): {avg_loss_domain_d:.4f} | "
            f"Adversarial Loss (F): {avg_loss_adversarial_f:.4f} | "
            f"MMD Loss: {avg_loss_mmd:.4f}"
        )
        history["loss_class"].append(avg_loss_class)
        history["loss_domain_d"].append(avg_loss_domain_d)
        history["loss_adversarial_f"].append(avg_loss_adversarial_f)
        history["loss_mmd"].append(avg_loss_mmd)

    print("Plotting training history...")
    plt.figure(figsize=(12, 8))
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        history["loss_class"],
        label="Classification Loss (F)",
        color=PLOT_COLORS[0],
    )
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        history["loss_domain_d"],
        label="Domain Loss (D)",
        color=PLOT_COLORS[1],
    )
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        history["loss_adversarial_f"],
        label="Adversarial Loss (F)",
        color=PLOT_COLORS[2],
    )
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        history["loss_mmd"],
        label="MMD Loss (F)",
        color=PLOT_COLORS[3],
    )
    plt.title("Training Loss Curves", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/training_history_dann.pdf")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
