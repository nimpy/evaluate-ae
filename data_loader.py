import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class PatchesDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image


def fetch_dataloader(types, data_dir, params, batch_size, rotation_deg=0, translation=0, scaling=1, shearing_deg=0):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    # define a training image loader that specifies transforms on images. See documentation for more details.
    train_transformer = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.RandomAffine(rotation_deg, translate=(translation, translation), scale=(1.0, scaling), shear=shearing_deg),
        # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
        transforms.ToTensor()])  # transform it into a torch tensor

    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor()])  # transform it into a torch tensor

    dataloaders = {}

    for split in ['train', 'validation', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split), "class0/")

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(PatchesDataset(path, train_transformer), batch_size=batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(PatchesDataset(path, eval_transformer), batch_size=batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders