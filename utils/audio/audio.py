import subprocess
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr


def convert_mp4_to_wav(file_path):
    new_file_path = file_path.replace('.mp4', '.wav')
    command = f'ffmpeg -i "{file_path}" -ab 160k -ac 1 -ar 44100 -vn "{new_file_path}"'
    code = subprocess.call(command, shell=True)
    if code != 0:
        print('=== Error during MP4->WAV conversion!')
    return code
    

def prepare_segments(file_path):
    sound = AudioSegment.from_file(file_path, format="wav")

    chunks = split_on_silence(
        sound,
        # split on silences longer than 1000ms (1 sec)
        min_silence_len=500,
        # anything under -25 dBFS is considered silence  // TODO: tweak that
        silence_thresh=-25, 
        # keep 200 ms of leading/trailing silence
        keep_silence=200
    )

    segments = []

    segment = AudioSegment.empty()
    for chunk in chunks:
        if segment.duration_seconds + chunk.duration_seconds < 60:
            segment += chunk
        else:
            segments.append(segment)
            segment = chunk
    segments.append(segment)

    return segments


def segments_to_text(segments):
    # Clear temp dir
    for file in os.listdir('tmp/'):
        os.remove('tmp/' + file)

    transcription = ''
    for i, segment in enumerate(segments):
        print(f'Preparing segment {i}')

        # Save segment as tmp file  TODO: Can somehow avoid this?
        segment.export(f'tmp/seg{i}.wav', format='wav')

        # Read audio and load into speech recognition module
        r = sr.Recognizer()
        with sr.AudioFile(f'tmp/seg{i}.wav') as source:
            audio = r.record(source) 

        # Run Google speech2text
        text = ''
        try:
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            text = r.recognize_google(audio, language='pl')
            transcription += f' {text}'
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return transcription

if __name__ == "__main__":
    file_path = '../../skyhacks_hackathon_dataset/audio_description/tarnowskie_gory_sztolnia_kopalnia_v2_30_10_2020_audiodeskrypcja_FINALNA.mp4'
    convert_mp4_to_wav(file_path)
    print('Starting Speech2Text service...')
    segments = prepare_segments(file_path.replace('.mp4', '.wav'))
    with open('text.txt', 'w') as file:
        file.write(segments_to_text(segments))
