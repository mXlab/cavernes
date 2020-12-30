from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, make_dataset
import torchvision.transforms.functional as F
from typing import Any, Callable, Optional
import os
import tqdm
from PIL import Image


class PrehistoryDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,):
        
        self.base_root = root
        root = os.path.join(root, "raw")
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.is_valid_file = is_valid_file

    def __getitem__(self, index):
        x, label = super().__getitem__(index)
        return x

    def resize(self, resolution: int, interpolation=Image.BILINEAR):
        root = os.path.join(self.base_root, str(resolution))
        for path, _ in tqdm.tqdm(self.imgs):
            new_path = path.replace("raw", str(resolution))
            if os.path.exists(new_path):
                continue
            img = self.loader(path)
            img = F.resize(img, resolution, interpolation)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            img.save(new_path)
      
        self.root = root
        self.samples = make_dataset(self.root, self.class_to_idx, self.extensions, self.is_valid_file)
        self.imgs = self.samples


