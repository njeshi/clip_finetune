import os
from copy import deepcopy
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
from pytorch_lightning import LightningDataModule


class FoodDataset(Dataset):
    """
    PyTorch Dataset instance to handle Food101 dataset
    source: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    """

    def __init__(self, img_dir, image_transform=None, prompt_transform=None, return_indices=False):
        """
        img_dir: str
            path to the directory with images
        image_transform: callable
            image preprocessing (augmentation, resize, cropping, etc)
        prompt_transform: callable
            function generating prompts for class names
        return_indices: bool
            whether return indices or not
        """
        self.img_dir = os.path.expanduser(img_dir)
        self.image_transform = image_transform
        self.prompt_transform = prompt_transform
        self.return_indices = return_indices

        self.class_to_idx = {}
        self.idx_to_class = {}
        self.paths_to_images = self._make_dataset()

    def _make_dataset(self):
        """ 
        Iterates over the image folder and assigns to each image the 
        class label and index
        """
        if self.return_indices:
            for idx, class_name in enumerate(os.scandir(self.img_dir)):
                class_name = class_name.name.replace('_', ' ')
                self.class_to_idx[class_name] = idx
            
        self.idx_to_class = {idx: class_name for class_name,
                             idx in self.class_to_idx.items()}

        paths_to_images = []
        for class_name in os.listdir(self.img_dir):
            for image_name in os.listdir(f'{self.img_dir}/{class_name}'):
                paths_to_images.append(
                    f"{self.img_dir}/{class_name}/{image_name}")
        return paths_to_images

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        """ Returns a pair of image and text prompt """

        img_path = self.paths_to_images[idx]

        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        prompt = img_path.split('/')[-2]
        prompt = prompt.replace('_', ' ')

        if self.image_transform:
            image = self.image_transform(image)

        if self.prompt_transform:
            prompt = self.prompt_transform(prompt)

        if self.return_indices:
            prompt = self.class_to_idx[prompt]

        return image, prompt

class FoodDataModule(LightningDataModule):
    """
    PyTorch Lighnting DataModule to setup and handle dataset split and loaders
    """
    def __init__(self, 
                folder: str,
                batch_size: str,
                image_transform = None):
        """
        Create a text image datamodule from directories with congruent text and image names.
        
        Parameters
        -----------
        folder: str
            Folder path containing all images
        batch_size: int 
            The batch size of each dataloader.
        """
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.dataset = FoodDataset(self.folder, image_transform, 
                                   return_indices = True)
        
    def setup(self, stage=None):
        train_idx, test_idx = train_test_split(np.arange(len(self.dataset)),
                                               test_size = 0.2,
                                               random_state=42)
        
        train_idx, val_idx = train_test_split(train_idx,
                                              test_size = 0.1,
                                              random_state=42)
        
        if stage in (None, "fit"):
            self.train_dataset = deepcopy(self.dataset)
            self.train_dataset.paths_to_images = [self.dataset.paths_to_images[idx] \
                                             for idx in train_idx]
            
            self.val_dataset = deepcopy(self.dataset)
            self.val_dataset.paths_to_images = [self.dataset.paths_to_images[idx] \
                                                for idx in val_idx]

        if stage in (None, "test"):
            self.test_dataset = deepcopy(self.dataset)
            self.test_dataset.paths_to_images = [self.dataset.paths_to_images[idx] \
                                                 for idx in test_idx]
            
            # Make sure test set knows all idxs and classes available
            self.test_dataset.idx_to_class = self.dataset.idx_to_class
            self.test_dataset.class_to_idx = self.dataset.class_to_idx
            
    
    def train_dataloader(self, sampler=None):
        # Shuffle is mutually exlusive with sampler
        shuffle = True if not sampler else False
        
        return DataLoader(self.train_dataset,
                          shuffle = shuffle,
                          sampler = sampler,
                          drop_last = True,
                          batch_size = self.batch_size, 
                          num_workers = os.cpu_count())
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          shuffle = False,
                          drop_last = True, 
                          batch_size = self.batch_size, 
                          num_workers = os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          shuffle = False,
                          drop_last = True, 
                          batch_size = self.batch_size, 
                          num_workers = os.cpu_count())

class KPerClassSampler(Sampler):
    """
    Sample k images from each class randomly
    """
    def __init__(self, dataset, k, seed):
        paths_to_images = dataset.paths_to_images
        grouped_by_class = self._group_by_class(paths_to_images)
        self.rng = np.random.default_rng(seed)
        self.sampled_images = self._sample_images(grouped_by_class, k)

    def _sample_images(self, grouped_by_class, k):
        sampled_img_paths = []
        for img_class, img_paths in grouped_by_class.items():
            replace = False if k<len(img_paths) else True
            
            sampled_img_paths.extend(
                self.rng.choice(img_paths, k, replace=replace)
                .tolist())
        return sampled_img_paths

    def _group_by_class(self, paths_to_images):
        grouped_by_class = defaultdict(list)
        for idx, img_path in enumerate(paths_to_images):
            # parse class name
            class_name = img_path.split('/')[-2]
            grouped_by_class[class_name].append((idx, img_path))
        return grouped_by_class

    def __len__(self):
        return len(self.sampled_images)

    def __iter__(self):
        self.rng.shuffle(self.sampled_images)
        return iter([int(idx) for idx, _ in self.sampled_images])
