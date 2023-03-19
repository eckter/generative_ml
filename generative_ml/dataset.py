from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class PokeDataset(Dataset):
    def __init__(self, data_dir: Path, size: int = 512):
        self._to_tensor = transforms.ToTensor()
        csv_path = data_dir / "pokemons.csv"
        images = data_dir / "images"
        df = pd.read_csv(csv_path)
        self.type_mapping = PokeDataset._make_type_mapping(df)
        self.images = PokeDataset._load_images(images, df["file_name"], size)
        self.types = self._load_types(df)
        assert len(self.images) == len(self.types)

    def __getitem__(self, index):
        return self.images[index], self.types[index]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_images(image_dir: Path, file_names: Iterable[str], size: int) -> torch.tensor:
        res = list()
        for file in file_names:
            res.append(PokeDataset._load_image(image_dir / file, size))
        return torch.stack(res)

    @staticmethod
    def _load_image(image_path: Path, size: int) -> torch.tensor:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.pad(image, (size, size), color="white")
        return transforms.ToTensor()(image)

    def _load_types(self, df: pd.DataFrame) -> torch.tensor:
        res = torch.zeros(len(df), len(self.type_mapping))
        for i, t in enumerate(df["type1"]):
            res[i][self.type_mapping[t]] = 1
        for i, t in enumerate(df["type2"]):
            if type(t) is str:
                res[i][self.type_mapping[t]] = 1
        return res

    @staticmethod
    def _make_type_mapping(df: pd.DataFrame) -> Mapping[str, int]:
        return {x: i for i, x in enumerate(df.type1.unique())}
