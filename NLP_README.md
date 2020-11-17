Install required libs:
```
conda install ffmpeg pydub nltk spacy
pip install -r requirements.txt
```

# Prepare PlWordNet

https://github.com/sebastiandziadzio/ivr-synsets
Last working version of Słowosieć: 2.3.7

First download `wordnet` corpus using interactive window (run from Python shell):
```
import nltk
nltk.download()
```

Then download Słowosieć (PlWordNet) from:
```
http://www.nlp.pwr.wroc.pl/plwordnet/download/plwordnet_2_3.7z
```

Overwrite the files from `*_pwn_format` dir (path could be checked in the NLTK download window) (tip: make backup of original files)

Download corpus for spacy:
```
python -m spacy download pl_core_news_sm
```

Example could be run using:
```
python nlp.py
```
