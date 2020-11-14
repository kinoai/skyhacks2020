import streamlit as st
import tempfile
from movie_to_frames import extract_frames
import altair as alt
import pandas as pd
from predict_example import predict
import cv2
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('training_labels.csv')
labels = [
    "Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross",
    "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape",
    "Mine", "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants",
    "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft",
    "Windows"
]


def frame_summary(frame, scores):
    st.image(frame, use_column_width=True)

    data = pd.DataFrame({
        'label': labels,
        'score': scores*100
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
    st.subheader("Result chart")
    st.write(bars)


def video_summary(dict, duration):
    predictions = np.round(dict['scores'])

    # print(predictions)

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_yticks(len(labels), labels, size='small')
    plt.yticks(range(len(labels)), labels, size='small')
    plt.xticks(range(predictions.shape[1]), range(predictions.shape[1]), size='small')
    plt.grid(color='k', linestyle='-', linewidth=1, axis='y', alpha=0.6)
    plt.tick_params(axis='y', which='major', labelsize=6)
    plt.xlabel("time [sec]")
    plt.ylabel("class")
    for i in range(1, predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i, j] == 1:
                if predictions[i-1, j] == 1:
                    plt.plot([i-1, i], [j, j], lw=4, c='orange')
                else:
                    plt.scatter(i, j, s=3, c='orange', marker='s')
    st.pyplot(fig)


def main():
    st.title("Online classifier")

    # uploaded file
    file = st.sidebar.file_uploader(label="Upload you MP4 file.",
                                    type=(["mp4"]))
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vc = cv2.VideoCapture(tfile.name)
        fps = int(vc.get(cv2.CAP_PROP_FPS))
        totalNoFrames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_in_seconds = float(totalNoFrames) / float(fps)
        res = {'frames': extract_frames(tfile.name, skip=fps)}
        res['scores'] = [predict(frame) for frame in res['frames']]
        st.video(file)
        f_idx = st.sidebar.slider("Select frame", 0, len(res['frames']) - 1)
        st.header("Selected frame")
        frame_summary(res['frames'][f_idx], res['scores'][f_idx])
        video_summary(res, duration_in_seconds)
    except AttributeError:
        st.write("Please upload your video!")


if __name__ == '__main__':
    main()
