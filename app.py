import streamlit as st
import tempfile
from movie_to_frames import extract_frames
from matplotlib import pyplot as plt
import numpy as np
import altair as alt
import pandas as pd

df = pd.read_csv('training_labels.csv')
labels = df.columns.tolist()[1:]


def frame_summary(frame):
    fig = plt.figure()
    plt.imshow(frame)
    plt.axis("off")
    st.pyplot(fig)

    pred = np.random.rand(len(labels)) * 100
    data = pd.DataFrame({
        'label': labels,
        'score': pred
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


def main():
    st.title("Online classifier")

    # uploaded file
    file = st.sidebar.file_uploader(label="Upload you MP4 file.",
                                    type=(["mp4"]))
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        frames = extract_frames(tfile.name, skip=100)
        # st.video(file)
        f_idx = st.sidebar.slider("Select frame", 0, len(frames) - 1)
        st.header("Selected frame")
        frame_summary(frames[f_idx])
    except AttributeError:
        st.write("Please upload your video!")


if __name__ == '__main__':
    main()
