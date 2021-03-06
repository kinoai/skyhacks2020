import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class SkyDatasetDescription:
    def __init__(self, description_file_path: str):
        self.description_file_path = description_file_path

        self.formats = ['JPG', 'JPEG', 'PNG']

    def get_description(self):
        with open(self.description_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            # skip header
            _ = reader.__next__()
            description = []

            for row in reader:
                file_name = row[0]
                *_, ext = file_name.split('.')
                if ext.upper() not in self.formats:
                    # print(file_name)
                    continue

                categories = list(map(lambda x: 1 if x == 'True' else 0, row[1:]))

                sample = {'name': file_name, 'label': categories}
                description.append(sample)

        return description


class SkyDataset(Dataset):
    def __init__(self, root_dir: str, dataset_description: SkyDatasetDescription, transforms=None):
        self.dataset_description = dataset_description.get_description()
        self.root_dir = root_dir
        self.trasforms = transforms

    def __getitem__(self, item):
        image_description = self.dataset_description[item]
        image_name = image_description['name']
        label = image_description['label']
        image = Image.open(os.path.join(self.root_dir, image_name))

        if self.trasforms:
            image = self.trasforms(image)

        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.dataset_description)


class SkyTestDataset(Dataset):
    def __init__(self, root_dir: str, transforms=None):
        self.file_paths = []
        self.filenames = []
        self.transforms = transforms
        self.formats = ['JPG', 'JPEG', 'PNG']

        for _, __, files in os.walk(root_dir):
            for file in files:
                self.file_paths.append(os.path.join(root_dir, file))
                self.filenames.append(file)

    def __getitem__(self, item):

        filename = self.file_paths[item]
        _, ext = filename.split('.')
        if ext.upper() not in self.formats:
            return None, self.filenames[item]

        image = Image.open(filename)

        try:
            if self.transforms:
                image = self.transforms(image)
        except Exception:
            return None, self.filenames[item]

        return image, self.filenames[item]
