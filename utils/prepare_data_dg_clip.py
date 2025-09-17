import clip
import torchvision.datasets as datasets
from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFilelist(torch.utils.data.Dataset):
    """
    Custom dataset that reads a file list.
    The file is expected to have lines in the format: `path/to/image.jpg label_index`.
    """
    def __init__(self, root, filelist_path, transform=None):
        self.root = root
        self.transform = transform
        self.img_paths = []
        self.labels = []

        with open(filelist_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.img_paths.append(os.path.join(self.root, path))
                # Correct the label from 1-indexed to 0-indexed
                self.labels.append(int(label) - 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Return 0 for text as placeholder, consistent with original structure
        return image, 0, label


class ImageTextData(object):

    def __init__(self, dataset_type, domain, root, preprocess, prompt=None):
        # Construct path based on the new directory structure
        domain_path = os.path.join(root, dataset_type, domain)
        if dataset_type == 'pacs':
            domain_path = os.path.join(root, dataset_type, 'raw_images', domain)

        data = datasets.ImageFolder(domain_path, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]

        self.preprocess = preprocess
        # self.text = clip.tokenize(self.labels) # This is handled by prompt learner now

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        # text_enc = self.text[label] # Not needed anymore
        return image, 0, label # Return 0 for text as placeholder

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



def get_data(data_name):
    datalist = {
        'pacs': 'pacs',
        'office_home': 'office_home',
        'vlcs': 'vlcs',
        'domain_net': 'domain_net'
    }
    if datalist[data_name] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getfeadataloader(args, model, dataset_type):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    np.random.seed(args.seed)
    for i, domain in enumerate(args.domains):
        data = ImageTextData(
            dataset_type, domain, args.root_dir, model.preprocess)
        l = len(data)
        index = np.arange(l)
        np.random.shuffle(index)

        # Corrected data split: 70% train, 10% val, 20% test
        train_end = int(l * 0.7)
        val_end = train_end + int(l * 0.1)

        trl.append(torch.utils.data.Subset(data, index[:train_end]))
        val.append(torch.utils.data.Subset(data, index[train_end:val_end]))
        tel.append(torch.utils.data.Subset(data, index[val_end:]))  # Use the rest for testing

        trd.append(torch.utils.data.DataLoader(
            trl[-1], batch_size=args.batch, shuffle=True, num_workers=16, pin_memory=True))
        vad.append(torch.utils.data.DataLoader(
            val[-1], batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))
        ted.append(torch.utils.data.DataLoader(
            tel[-1], batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))
    return trd, vad, ted



def pacs(args, model):
    """
    Loads PACS data using pre-defined split files.
    """
    trd, vad, ted = [], [], []
    dataset_root = os.path.join(args.root_dir, 'pacs', 'raw_images')
    splits_dir = os.path.join(args.root_dir, 'pacs', 'Train val splits and h5py files pre-read')

    for domain in args.domains:
        train_file = os.path.join(splits_dir, f"{domain}_train_kfold.txt")
        val_file = os.path.join(splits_dir, f"{domain}_crossval_kfold.txt")
        test_file = os.path.join(splits_dir, f"{domain}_test_kfold.txt")

        train_dataset = ImageFilelist(dataset_root, train_file, transform=model.preprocess)
        val_dataset = ImageFilelist(dataset_root, val_file, transform=model.preprocess)
        test_dataset = ImageFilelist(dataset_root, test_file, transform=model.preprocess)

        trd.append(torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, num_workers=16, pin_memory=True))
        vad.append(torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))
        ted.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))

    return trd, vad, ted


def office_home(args, model):
    return getfeadataloader(args, model, 'office_home')


def vlcs(args, model):
    return getfeadataloader(args, model, 'vlcs')


def domain_net(args, model):
    """
    Loads DomainNet data using pre-defined split files.
    A validation set is created from the training set.
    """
    trd, vad, ted = [], [], []
    dataset_root = os.path.join(args.root_dir, 'domain_net')

    for domain in args.domains:
        train_file = os.path.join(dataset_root, f"{domain}_train.txt")
        test_file = os.path.join(dataset_root, f"{domain}_test.txt")

        # Create train and validation sets from the train_file
        full_train_dataset = ImageFilelist(dataset_root, train_file, transform=model.preprocess)
        test_dataset = ImageFilelist(dataset_root, test_file, transform=model.preprocess)

        # Split training data into new train and validation sets (90%/10%)
        np.random.seed(args.seed)
        indices = np.arange(len(full_train_dataset))
        np.random.shuffle(indices)
        val_size = int(len(full_train_dataset) * 0.1)
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

        trd.append(torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch, shuffle=True, num_workers=16, pin_memory=True))
        vad.append(torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))
        ted.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch, shuffle=False, num_workers=16, pin_memory=True))

    return trd, vad, ted


