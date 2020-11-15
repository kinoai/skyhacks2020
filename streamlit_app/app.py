import streamlit as st
import tempfile
from movie_to_frames import extract_frames
import altair as alt
import pandas as pd
from predict_example import predict
from audio import convert_mp4_to_wav, prepare_segments, segments_to_text
from nlp import lemmize_text, detect_labels
import cv2
import numpy as np
import matplotlib.pyplot as plt

labels = [
    "Amusement park", "Animals", "Bench", "Building", "Castle", "Cave",
    "Church", "City", "Cross",
    "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass",
    "Graveyard", "Lake", "Landscape",
    "Mine", "Monument", "Motor vehicle", "Mountains", "Museum",
    "Open-air museum", "Park", "Person", "Plants",
    "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility",
    "Stairs", "Trees", "Watercraft",
    "Windows"
]

words = {'building': {'strop': 1}, 'castle': {'zamek': 8}, 'cave': {'strop': 1}}


@st.cache
def load_data(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    convert_mp4_to_wav(tfile.name)
    segments = prepare_segments(tfile.name + '.wav')
    text = segments_to_text(segments)
    text = nlp.lemmize_text(text)

    labels = nlp.detect_labels(text)
    print(f"labels-------\n{labels}")

    vc = cv2.VideoCapture(tfile.name)
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    # totalNoFrames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    # duration_in_streamseconds = float(totalNoFrames) / float(fps)
    res = {'frames': extract_frames(tfile.name, skip=fps)}
    res['scores'] = [predict(frame) for frame in res['frames']]
    res['labels'] = labels
    return res, fps


def frame_summary(frame, scores):
    st.image(frame, use_column_width=True)

    data = pd.DataFrame({
        'label': labels,
        'score': scores * 100
    })
    mask = data['score'] > 50
    data = data[mask]
    data = data.sort_values(by=['score'], ascending=False)
    bars = (
        alt.Chart(data)
            .properties(height=600, width=650)
            .mark_bar()
            .encode(x=alt.X("label:O", sort=data["label"].tolist()),
                    y="score:Q")
    )
    return bars


def video_summary(predictions):
    predictions = np.round(predictions)

    fig = plt.figure()
    plt.yticks(range(len(labels)), labels, size='small')
    plt.xticks(range(predictions.shape[1]), range(predictions.shape[1]),
               size='small')
    plt.grid(color='k', linestyle='-', linewidth=1, axis='y', alpha=0.6)
    plt.tick_params(axis='both', which='both', labelsize=6)
    plt.xlabel("time [sec]")
    plt.ylabel("class")
    for i in range(1, predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i, j] == 1:
                if predictions[i - 1, j] == 1:
                    plt.plot([i - 1, i], [j, j], lw=5, c='#4C78A8')
                else:
                    plt.scatter(i, j, s=7, c='#4C78A8', marker='s')
    return fig


def text_summary(labels):
    print(labels)


def main():
    st.title("Online classifier")

    # uploaded file
    file = st.sidebar.file_uploader(label="Upload you MP4 file.",
                                    type=(["mp4"]))
    if file is not None:
        res, fps = load_data(file)
        st.header("Selected video")
        st.video(file)
        f_idx = st.sidebar.slider("Select frame", 0, len(res['frames']) - 1)
        st.header("Selected frame")
        bars = frame_summary(res['frames'][f_idx], res['scores'][f_idx])
        st.subheader("Frame result chart")
        st.write(bars)
        st.subheader("Chart of the time intervals in which particular classes "
                     "appears on the video")
        fig = video_summary(res['scores'])
        st.pyplot(fig)

        st.header("Audio labels detection")
        st.write(text_summary(res['labels']))

    else:
        st.write("Please upload your video!")


if __name__ == '__main__':
    main()
