from pytorch_lightning import LightningModule
import csv


# field names
fields = [
    "Name", "Amusement park",	"Animals Bench", "Building", "Castle", "Cave", "Church", "City", "Cross",
    "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape",
    "Mine", "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants",
    "Reservoir",	"River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft",
    "Windows"
]


# data rows of csv file
rows = []


pretrained_model = LightningModule.load_from_checkpoint("skyhacks2020/1n3uwxgu/checkpoints/epoch=2.ckpt")
pretrained_model.freeze()


# out = pretrained_model(x)
out = None


# name of csv file
filename = "task1_answers.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)


