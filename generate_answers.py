import torch
from utils.lightning_wrapper import LitModel
from utils.datasets import SkyTestDataset
from utils.transform import test_preprocess
from main import load_config
import csv


# field names
fields = [
    "Name", "Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross",
    "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape",
    "Mine", "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants",
    "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft",
    "Windows"
]

config = load_config()
# pretrained_model = LitModel.load_from_checkpoint("skyhacks2020/1brh2881/checkpoints/epoch=8.ckpt", config=config)
pretrained_model = LitModel.load_from_checkpoint("example.ckpt", config=config)
pretrained_model.freeze()

test_dataset = SkyTestDataset(root_dir="skyhacks_hackathon_dataset/live_test_images", transforms=test_preprocess)

# data rows of csv file
answers = []
for img, filename in test_dataset:

    if img is None:
        print(filename)
        zeros = [0 for x in range(len(fields)-1)]
        answers.append([filename] + zeros)
        continue

    img = img.reshape((1, 3, 224, 224))
    logits = pretrained_model(img)
    logits = logits.squeeze()
    preds = torch.where(logits > 0.5, 1, 0).tolist()
    answers.append([filename] + preds)


def sortFunc(e):
    return e[0]


answers.sort(key=sortFunc)
print(answers)

# name of csv file
filename = "task1_answers.csv"

# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(answers)
