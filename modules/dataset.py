import torch
from torch.utils.data import DataLoader, Dataset
import  torchvision.transforms as T
import torchvision.datasets as datasets

class Simclrtransform:
    def __init__(self,image_size = 32):
        self.transform = T.Compose([
            T.RandomResizedCrop(size = image_size),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)],p = 0.8),
            T.RandomGrayscale(p = 0.2),
            T.GaussianBlur(kernel_size= 3),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    def __call__(self,x):
        return self.transform(x) ,self.transform(x)
    
# Simclr dataset
class simclrDataset:
    def __init__(self,root = "data",train = True,download = True):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, transform = Simclrtransform(), download=download
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        (x1, x2), _ = self.dataset[idx]  # ignore label for contrastive
        return x1, x2
        pass
def get_dataloader(batch_size=256, num_workers=2):
    train_dataset = simclrDataset(train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    return train_loader


