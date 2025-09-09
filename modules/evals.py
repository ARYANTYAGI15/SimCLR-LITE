# modules/eval.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from modules.dataset import get_dataloader


def linear_eval(backbone, epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    """
    Linear evaluation protocol for SimCLR.
    Freeze backbone and train a linear classifier on CIFAR-10.
    """
    # Get CIFAR-10 dataloaders (with labels)
    train_loader = get_dataloader(batch_size=batch_size, supervised=True, train=True)
    test_loader = get_dataloader(batch_size=batch_size, supervised=True, train=False)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Simple linear classifier (input = 128, output = 10 classes)
    classifier = nn.Linear(128, 10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward through frozen backbone
            with torch.no_grad():
                feats = backbone(imgs, return_features=True)

            # Forward through classifier
            outputs = classifier(feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    # Evaluation
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = backbone(imgs, return_features=True)
            outputs = classifier(feats)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = 100. * correct / total
    print(f"✅ Linear Eval Accuracy on CIFAR-10: {test_acc:.2f}%")

    return classifier, test_acc
