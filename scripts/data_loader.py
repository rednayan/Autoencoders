import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
import warnings


class SportBallsDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders_sportballs(batch_size, num_workers=0, train_transforms=None, test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    train_dataset = SportBallsDataset(root_dir='SportBallsSmall/Train',
                                      labels_file='SportBallsSmall/Train/labels.csv',
                                      transform=train_transforms)

    test_dataset = SportBallsDataset(root_dir='SportBallsSmall/Test',
                                     labels_file='SportBallsSmall/Test/labels.csv',
                                     transform=test_transforms)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    
    return train_loader, test_loader