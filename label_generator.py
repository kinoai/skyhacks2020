import csv
import os
from PIL import Image
from torch.utils.data import Dataset


class NewData(Dataset):
    def __init__(self, root_dir: str, transforms=None):
        self.file_paths = []
        self.filenames = []
        self.transforms = transforms
        self.formats = ['JPG', 'JPEG', 'PNG']

        for _, __, files in os.walk(root_dir):
            for file in files:
                _, ext = file.split('.')
                if ext.upper() not in self.formats:
                    continue

                self.file_paths.append(os.path.join(root_dir, file))
                self.filenames.append(file)

    def __getitem__(self, item):

        filename = self.file_paths[item]

        image = Image.open(filename)

        if self.transforms:
            image = self.transforms(image)

        return image, self.filenames[item]


# field names
fields = [
    "Name", "Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross",
    "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape",
    "Mine", "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants",
    "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft",
    "Windows"
]


new_data = NewData(root_dir="new_data/")
label_data = []
for img, filename in new_data:
    label_data.append([filename] + [0 for i in range(len(fields)-1)])

index = 7
for x in label_data:
    x[index] = 1


def sortFunc(e):
    return e[0]


label_data.sort(key=sortFunc)
print(label_data)

# name of csv file
filename = "new_labels.csv"


# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(label_data)
