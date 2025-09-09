# modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device="cpu"):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        # Mask to exclude self-comparisons
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # zero out self-similarity
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: two batches of embeddings (batch_size, dim)
        """
        N = 2 * self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # (2N, dim)

        # normalize
        z = F.normalize(z, dim=1)

        # cosine similarity (2N x 2N)
        sim = torch.matmul(z, z.T) / self.temperature

        # remove self-comparisons
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        negatives = sim[self.mask].view(N, -1)

        # labels (positives are at index 0 for each sample)
        labels = torch.zeros(N).long().to(self.device)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        return loss / N
