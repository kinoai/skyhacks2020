class_names = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

yolo_to_sky_mapper = {
    14: 1,  # animals
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    13: 2,  # bench
    40: 10,  # food
    41: 10,
    42: 10,
    43: 10,
    44: 10,
    45: 10,
    46: 10,
    47: 10,
    48: 10,
    49: 10,
    50: 10,
    51: 10,
    52: 10,
    53: 10,
    54: 10,
    55: 10,
    56: 13,  # furniture
    57: 13,
    59: 13,
    60: 13,
    2: 20,  # motor vehicle
    3: 20,
    4: 20,
    5: 20,
    6: 20,
    7: 20,
    8: 20,
    0: 25,  # person
    58: 26,  # plants
    29: 32,  # sport
    30: 32,
    31: 32,
    32: 32,
    33: 32,
    34: 32,
    35: 32,
    36: 32,
    37: 32,
    38: 32,
}

to_be_replaced = [2, 25, 13, 20]
