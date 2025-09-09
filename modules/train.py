# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from modules.model import get_model
from modules.projections import ProjectionHead
from modules.losses import NTXentLoss
from modules.dataset import get_dataloader

def train_simclr(epochs=10, batch_size=256, lr=1e-3, temperature=0.5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # backbone + projection head
    backbone = get_model(num_classes=10, device=device)
    proj_head = ProjectionHead(input_dim=128, hidden_dim=128, output_dim=64).to(device)


    # dataloader
    train_loader = get_dataloader(batch_size=batch_size)

    # optimizer
    optimizer = optim.Adam(list(backbone.parameters()) + list(proj_head.parameters()), lr=lr)

    # loss
    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature, device=device)

    backbone.train()
    proj_head.train()

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            # forward
            h1 = backbone(x1, return_features=True)
            h2 = backbone(x2, return_features=True)

            z1 = proj_head(h1)
            z2 = proj_head(h2)

            # loss
            loss = criterion(z1, z2)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)

    # save model
    torch.save({
        "backbone": backbone.state_dict(),
        "proj_head": proj_head.state_dict()
    }, "checkpoints/simclr.pth")

    print("✅ Training complete. Model saved at checkpoints/simclr.pth")
