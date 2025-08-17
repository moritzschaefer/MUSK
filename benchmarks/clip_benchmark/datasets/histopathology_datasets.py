import os
import glob
import random

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import h5py
import numpy
import json
import pandas as pd
from pathlib import Path
import io
import logging
logger = logging.getLogger(__name__)


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=True, val=False,
                 tumor=False, scale_factors=[0.6, 1.0, 1.0]):  # scale_factors are approximates to match our training (we don't have zoom out)
        # store scale factors for multi-scale cropping
        self.scale_factors = scale_factors

        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)
        self.data_root = root

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train

        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.cat_to_num_map) if not self.tumor else list(self.tumor_map)

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def _crop_tile(self, image, scale_factor):
        """Return a centered square crop of the PIL Image according to scale_factor.
        If scale_factor == 1.0 the original image is returned.
        For scale_factor < 1.0 a centered square crop with that fraction of width/height is returned.
        """
        if scale_factor == 1.0:
            return image

        w, h = image.size
        cw = int(round(w * scale_factor))
        ch = int(round(h * scale_factor))
        side = min(cw, ch)

        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        return image.crop((left, top, left + side, top + side))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')

        multi_scale_tiles = []

        for scale_factor in self.scale_factors:
            try:
                tile = self._crop_tile(image, scale_factor)

                if self.transform.__class__.__name__ == "CLIPProcessor":
                    t = self.transform(images=tile, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    t = self.transform(tile)

            except Exception as e:
                logger.error(
                    f"Error processing tile for image {image_path} at scale {scale_factor}: {e}"
                )
                t = torch.zeros((3, 224, 224), dtype=torch.float32)

            multi_scale_tiles.append(t)

        # Stack the scales for this sample: (n_scales, 3, 224, 224)
        multi_scale_patch = torch.stack(multi_scale_tiles, dim=0)

        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return multi_scale_patch, self.labels[index]


class PannukeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, scale_factors=[0.6, 1.0, 1.0]):
        self.root = root

        df = pd.read_csv(os.path.join(root, "PanNuke_all_binary.csv"))
        self.df = df[df['split'] == 'train'] if train else df[df['split'] == 'test']

        self.transform = transform
        self.scale_factors = scale_factors

        self.classes = ["benign",
                        "malignant"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def _crop_tile(self, image, scale_factor):
        """Return a centered square crop of the PIL Image according to scale_factor.
        If scale_factor == 1.0 the original image is returned.
        """
        if scale_factor == 1.0:
            return image

        w, h = image.size
        cw = int(round(w * scale_factor))
        ch = int(round(h * scale_factor))
        side = min(cw, ch)

        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        return image.crop((left, top, left + side, top + side))

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.df.iloc[index]['image'])
        image = Image.open(fpath).convert("RGB")

        multi_scale_tiles = []
        for scale_factor in self.scale_factors:
            try:
                tile = self._crop_tile(image, scale_factor)

                if self.transform.__class__.__name__ == "CLIPProcessor":
                    t = self.transform(images=tile, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    t = self.transform(tile)

            except Exception as e:
                logger.error(
                    f"Error processing tile for file {fpath} at scale {scale_factor}: {e}"
                )
                t = torch.zeros((3, 224, 224), dtype=torch.float32)

            multi_scale_tiles.append(t)

        multi_scale_patch = torch.stack(multi_scale_tiles, dim=0)

        label = 1 if 'malignant' in self.df.iloc[index]['caption'] else 0
        return multi_scale_patch, self.df.iloc[index]['caption']



class UnitopathoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "images_train.json")))
        else:
            self.data = json.load(open(os.path.join(root, "images_test.json")))
        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}
        # NORM - Normal
        # tissue;
        # HP - Hyperplastic
        # Polyp;
        # TA.HG - Tubular
        # Adenoma, High - Grade
        # dysplasia;
        # TA.LG - Tubular
        # Adenoma, Low - Grade
        # dysplasia;
        # TVA.HG - Tubulo - Villous
        # Adenoma, High - Grade
        # dysplasia;
        # TVA.LG - Tubulo - Villous
        # Adenoma, Low - Grade
        # dysplasia.

        self.classes = ["Hyperplastic Polyp",
                        "Normal tissue",
                        "Tubular Adenoma, High-Grade dysplasia",
                        "Tubular Adenoma, Low-Grade dysplasia",
                        "Tubulo-Villous Adenoma, High-Grade dysplasia",
                        "Tubulo-Villous Adenoma, Low-Grade dysplasia"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class UnitopathoRetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset for unitopatho image retrieval, using all samples.
    """

    def __init__(self, root, transform=None, train=True):
        self.data = json.load(open(os.path.join(root, "images.json")))

        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}

        # these prompts work better!
        self.classes = ["HP",
                        "NORM",
                        "TA.HG",
                        "TA.LG",
                        "TVA.HG",
                        "TVA.LG"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label




class PathMMUDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Initialize the dataset by processing the JSON files and setting up the image roots.
        """
        self.transform = transform
        self.root = Path(root)
        self.items = []

        # Process subsets
        self.img_root = {
            "Book": self.root / "Book/images",
            "WebPathology": self.root / "WebPathology/images",
            "Twitter": self.root / "Twitter/images",
        }

        self._process_json_files(self.root / "Book")
        self._process_json_files(self.root / "WebPathology")
        self._process_json_files(self.root / "Twitter")

    def _process_json_files(self, dir_path):
        """
        Process all JSON files in the given directory and extend the items list.
        """
        dir_path = dir_path / "GPTCaption"
        json_files = dir_path.glob("*.json")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    
                    data_list = list(data.values())
                    dataset_name = json_file.parts[-3]
                    for entry in data_list:
                        entry['dataset'] = dataset_name

                    self.items.extend(data_list)

                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {json_file}")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve an item by index, including the image and its corresponding caption.
        """
        
        item = self.items[idx]
        img_root = self.img_root[item['dataset']]
        img_path = img_root / item['img_path']

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None
        
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            image = self.transform(image)
        
        return image, item['caption']

