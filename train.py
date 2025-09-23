import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dann import DANN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOURCE_IMG_DIR = "./spectrograms/source"
TARGET_IMG_DIR = "./spectrograms/target"
MODEL_SAVE_PATH = "./dann_model.pth"

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


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
)

target_loader = DataLoader(
    dataset=target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(
        f"Source dataset size: {len(source_dataset)}, classes: {source_dataset.classes}"
    )
    print(f"Target dataset size: {len(target_dataset)}")

    model = DANN(num_classes=len(source_dataset.classes)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        len_dataloader = min(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for i in tqdm(range(len_dataloader), desc=f"Epoch {epoch + 1} / {NUM_EPOCHS}"):
            source_data, source_label = next(source_iter)
            target_data, _ = next(target_iter)
            source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
            target_data = target_data.to(DEVICE)

            # 0 for source, 1 for target
            domain_label_source = torch.zeros(source_data.size(0)).long().to(DEVICE)
            domain_label_target = torch.ones(target_data.size(0)).long().to(DEVICE)

            p = float(i + epoch * len_dataloader) / (NUM_EPOCHS * len_dataloader)
            alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10 * p))) - 1
            optimizer.zero_grad()

            class_output, domain_output_source = model(source_data, alpha)
            err_s_label = loss_domain(class_output, source_label)
            err_s_domain = loss_domain(domain_output_source, domain_label_source)

            _, domain_output_target = model(target_data, alpha)
            err_t_domain = loss_domain(domain_output_target, domain_label_target)

            err_domain = err_s_domain + err_t_domain
            total_loss = err_s_label + err_domain

            total_loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1} / {NUM_EPOCHS}] - ",
            f"Total Loss: {total_loss.item():.4f}, ",  # type: ignore
            f"Class Loss: {err_s_label.item():.4f}, ",  # type: ignore
            f"Domain Loss: {err_domain.item():.4f}, ",  # type: ignore
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
