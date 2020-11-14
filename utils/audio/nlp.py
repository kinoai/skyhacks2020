from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('pl_core_news_sm')

LABELS_EN = [
    'amusement park', 'animals', 'bench', 'building', 'castle', 'cave', 'church', 'city', 'cross', 'cultural institution', 
    'food', 'footpath', 'forest', 'furniture', 'grass', 'graveyard', 'lake', 'landscape', 'mine', 'monument', 'motor vehicle', 
    'mountains', 'museum', 'open-air museum', 'park', 'person', 'plants', 'reservoir', 'river', 'road', 'rocks', 'snow', 'sport', 
    'sports facility', 'stairs', 'trees', 'watercraft', 'windows']

LABELS_PL = [
    'park rozrywki', 'zwierzę', 'ławka', 'budynek', 'zamek', 'jaskinia', 'kościół', 'miasto', 'krzyż', 'instytucja kultury',
    'jedzenie', 'chodnik', 'las', 'mebel', 'trawa', 'cmentarz', 'jezioro', 'krajobraz', 'kopalnia', 'pomnik', 'pojazd silnikowy',
    'góry', 'muzeum', 'skansen', 'park', 'osoba', 'roślina', 'zbiornik wodny', 'rzeka', 'droga', 'kamień', 'śnieg', 'sport',
    'obiekt sportowy', 'schody', 'drzewo', 'statek', 'okno'
]


def flatten(deep_list):
    return [e[0] for e in deep_list]


def find_related_words(word, syn_id=0):
    base_synset = wn.synsets(word)
    if not base_synset:
        return []
    base_synset = base_synset[syn_id]
    lemmas = base_synset.lemma_names()
    
    # hyponyms = [ss.lemma_names() for ss in base_synset.hyponyms()]
    hypernyms = flatten([ss.lemma_names() for ss in base_synset.hypernyms()])
    part_holonyms = flatten([ss.lemma_names() for ss in base_synset.part_holonyms()])
    part_meronyms = flatten([ss.lemma_names() for ss in base_synset.part_meronyms()])
    entailments = flatten([ss.lemma_names() for ss in base_synset.entailments()])
    related = [ss.name() for ss in base_synset.lemmas()[0].derivationally_related_forms()]

    return [*lemmas, *hypernyms, *part_holonyms, *part_meronyms, *entailments, *related]


def prepare_labels():
    labels = {}
    for word_pl, word_en in zip(LABELS_PL, LABELS_EN):
        related = [e.replace('_', ' ') for e in find_related_words(word_pl.replace(' ', '_'))]
        labels[word_en] = related

    lemmized_labels = {}
    for label in labels:
        lemmized_labels[label] = [' '.join([t.lemma_ for t in nlp(l)]) for l in labels[label]]
        
    return labels


def lemmize_text(text):
    return ' '.join([token.lemma_ for token in nlp(text)])


def detect_labels(text):
    lemmized_labels = prepare_labels()
    lemmized_text = lemmize_text(text)

    detected = {}
    for key, value in lemmized_labels.items():
        for word in value:
            if word in lemmized_text:
                if key not in detected.keys():
                    detected[key] = {}
                if word not in detected[key].keys():
                    detected[key][word] = 0
                detected[key][word] = lemmized_text.count(word)
    
    return detected


if __name__ == "__main__":
    text = open('example_transcription.txt').read()
    text = lemmize_text(text)
    labels = detect_labels(text)
    print(labels)
