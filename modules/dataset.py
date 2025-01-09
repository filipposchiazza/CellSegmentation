import os
import random
import torch
import torch.utils.data as data
from torchvision.io import read_image


class CitoDataset(data.Dataset):

    def __init__(self, dir, transform=None, fraction=1.0):
        """Citology dataset for segmentation: image + mask.

        Parameters
        ----------
        dir : str
            Path to the directory containing images.
        transform : torchvision.transforms
            Transformation to apply to the images.
        fraction : float
            Fraction of the dataset to keep.
        """

        self.transform = transform
        self.fraction = fraction
        self.dir = dir
        
        # List all images in the directories
        self.images = [os.path.join(dir, img) for img in sorted(os.listdir(dir))]
            
        # shuffle the list, after setting the seed
        random.seed(42)
        random.shuffle(self.images)

        # Keep only a fraction of the dataset
        self.images = self.images[:int(len(self.images) * self.fraction)]


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        full_img = read_image(self.images[idx])[:3, :, :] # image with additional mask 
        img = full_img[:, :, :full_img.shape[2] // 2] / 255.0
        mask = full_img[0, :, full_img.shape[2] // 2:] / 255.0
        return img, mask.unsqueeze(0)
        


def prepare_ImageDataset(img_dir, 
                         batch_size,
                         validation_split,
                         transform=None,
                         seed=123, 
                         fraction=1.0):
    """Prepare the image dataset for training and validation.

    Parameters
    ----------
    img_dir : str
        Path to the directory containing images.
    batch_size : int
        Size of the batch.
    validation_split : float
        Fraction of the dataset to use for validation.
    transform : torchvision.transforms
        Transformation to apply to the images.
    seed : int
        Seed for the random generator.
    fraction : float
        Fraction of the dataset to keep.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
        Dataset for training.
    val_dataset : torch.utils.data.Dataset
        Dataset for validation.
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training.
    val_dataloader : torch.utils.data.DataLoader
        Dataloader for validation.
    """
    dataset = CitoDataset(img_dir, transform=transform, fraction=fraction)

    val_len = int(len(dataset) * validation_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = data.random_split(dataset, [train_len, val_len], generator=generator)
    
    train_dataloader = data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=4)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4)
    
    return train_dataset, val_dataset, train_dataloader, val_dataloader