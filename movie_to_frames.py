from decord import VideoReader, cpu
from PIL import Image


def extract_frames(path, skip=25):
    # with open(path, 'rb') as f:
    vr = VideoReader(path, ctx=cpu(0))
    # (batch_size, height, width, channels)
    batch = vr.get_batch([range(0, len(vr), skip)]).asnumpy()
    frames = [Image.fromarray(frame) for frame in batch]
    return frames
