from sky_utils.lightning_wrapper import LitModel
from sky_utils.transform import test_preprocess
from main import load_config
import torch

config = load_config()
pretrained_model = LitModel.load_from_checkpoint("epoch=27.ckpt", config=config)
pretrained_model.freeze()


def predict(img):
    img = test_preprocess(img)
    img = img.reshape((1, 3, 224, 224))
    logits = pretrained_model(img)
    logits = logits.squeeze().numpy()
    # preds = torch.where(logits > 0.5, 1, 0).tolist()

    return logits
