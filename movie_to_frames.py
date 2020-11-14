from decord import VideoReader, cpu


def extract_frames(path, skip=25):
    with open(path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    # (batch_size, height, width, channels)
    frames = vr.get_batch([range(0, len(vr), skip)]).asnumpy()
    # return np.swapaxes(frames, 1, 3)
    return frames

# path = r"C:\Users\kacwl\Downloads\test.mp4"

# frames = extract_frames(path)
# print(frames.shape)
