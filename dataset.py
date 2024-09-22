from torch.utils.data import Dataset


# Custom Dataset class to apply transforms
class CelebADataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]

        if len(image.getbands()) != 3:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
