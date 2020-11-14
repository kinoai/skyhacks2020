from utils.lightning_wrapper import LitModel
from utils.transform import test_preprocess
from main import load_config
import torch


config = load_config()
pretrained_model = LitModel.load_from_checkpoint("example.ckpt", config=config)
pretrained_model.freeze()


img = None  # tu wczytac


img = test_preprocess(img)
img = img.reshape((1, 3, 224, 224))
logits = pretrained_model(img)
logits = logits.squeeze()
preds = torch.where(logits > 0.5, 1, 0).tolist()

print(preds)
