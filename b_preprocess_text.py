import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import re
import unicodedata
from utils.constants import TextElements, valid_words
from multiprocessing import Pool
import pandas as pd
import spacy
from utils.utils import re_expressions, create_stopwords
import textdescriptives as td
import textstat
import text2emotion as te
import enchant
from spacytextblob.spacytextblob import SpacyTextBlob

re_stopwords = re_expressions(create_stopwords())

nlp = spacy.load('en_core_web_md', disable=["ner", "senter"])
nlp.add_pipe('textdescriptives')
nlp.add_pipe('spacytextblob')


def lemmatization_text(text):
    original_text = text
    result = {}
    doc = nlp(text)
    text_lemmatizer = ' '.join(e.lemma_ for e in doc)
    text_lemmatizer = re.sub(re_stopwords, " ", text_lemmatizer, flags=re.IGNORECASE)
    text_lemmatizer = re.sub(r"\s+", " ", text_lemmatizer)

    result["text_preprocessed"] = text_lemmatizer
    result["polarity"] = doc._.polarity
    result["pos_prop_PRON"] = td.extract_df(doc).iloc[0].get("pos_prop_PRON")
    result["pos_prop_AUX"] = td.extract_df(doc).iloc[0].get("pos_prop_AUX")
    result["pos_prop_DET"] = td.extract_df(doc).iloc[0].get("pos_prop_DET")
    result["pos_prop_ADJ"] = td.extract_df(doc).iloc[0].get("pos_prop_ADJ")
    result["pos_prop_NOUN"] = td.extract_df(doc).iloc[0].get("pos_prop_NOUN")
    result["pos_prop_PUNCT"] = td.extract_df(doc).iloc[0].get("pos_prop_PUNCT")

    result["flesch_reading_ease"] = textstat.flesch_reading_ease(original_text)
    result["mcalpine_eflaw"] = textstat.mcalpine_eflaw(original_text)
    result["reading_time"] = textstat.reading_time(original_text)
    result["difficult_words_count"] = textstat.difficult_words(original_text)
    result["word_count"] = textstat.lexicon_count(original_text)

    out_emotion = te.get_emotion(original_text)
    result["Angry"] = out_emotion["Angry"]
    result["Fear"] = out_emotion["Fear"]
    result["Happy"] = out_emotion["Happy"]
    result["Sad"] = out_emotion["Sad"]
    result["Surprise"] = out_emotion["Surprise"]

    return result


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode('utf-8')


def split_hashtag_to_words_all_possibilities(hashtag):
    hashtag = hashtag.lower()
    all_possibilities = []

    if hashtag in word_dictionary:
        return [[hashtag]]
    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)

            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
    return all_possibilities


def split_hashtag_text(text):
    list_words = []
    for z in text.split():
        if z.startswith("@") or z.startswith("#"):
            z = z[1:]
            z = re.sub(TextElements.re_list_chars_to_remove, " ", z, flags=re.IGNORECASE)
            z = re.sub(r"\d+", " ", z)
            z = z.rstrip().strip()
            z = re.sub(r"\s+", " ", z)
            list_elements = z.split()

            final_list = []
            for element in list_elements:
                if element != "" and denchant.check(element):
                    if len(element) > 1:
                        final_list.append(element)
                elif element.islower() or element.isupper() or element.istitle():
                    if element.isupper():
                        element = element.lower()
                    list_split = split_hashtag_to_words_all_possibilities(element)

                    if len(list_split) > 0:
                        new_hashtag = merge_hashtag(list_split[0])
                        if len(element) > 1:
                            final_list.append(new_hashtag)
                    else:
                        if len(element) > 1:
                            final_list.append(element)
                else:
                    list_split = re.split('(?=[A-Z])', element)
                    new_hashtag = merge_hashtag(list_split)
                    if len(element) > 1:
                        final_list.append(new_hashtag)

            final_text = ' '.join(final_list)
            list_words.append(final_text)
        else:
            list_words.append(z)

    text_lemmatizer = ' '.join(list_words)
    return text_lemmatizer


def preprocess_and_remove_people_in_file(text):
    text = re.sub(
        r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        "", text)
    text = re.sub(r"\b[\w|\.||=-]+@[\w|\.|-]+\b", "", text)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub(r"(?:(pic.|http|www|\w+)?\:(//)*)\S+", " ", text)
    text = split_hashtag_text(text)

    result = lemmatization_text(text)
    text = result["text_preprocessed"]

    text = re.sub(r"(\*|\[|\]|=|\(|\)|\$|\"|\}|\{|\||\+|&|€|£|/|º)+", " ", text)
    text = re.sub(r"(\s|\t|\\n|\n)+", " ", text)
    text = re.sub(r"[\,|\.|'|:|;|\-|–]+", " ", text)
    text = re.sub(r"\d+[A-Za-z]*", " ", text)

    text = remove_accents(text)

    text = re.sub(TextElements.re_list_chars_to_remove, " ", text, flags=re.IGNORECASE)

    list_words = re.findall(r"\b[a-zA-Z\-]{2,}\b", text)
    text = " ".join(list_words)
    text = re.sub(r"\s+", " ", text)

    text = text.lower()
    result["text_preprocessed"] = text

    return result


def merge_hashtag(list_splited):
    new_hashtag = ""
    enter_in_upper_case = True
    for word in list_splited:

        if len(word) <= 1:
            if word != "":
                if enter_in_upper_case:
                    new_hashtag = new_hashtag + word
                else:
                    enter_in_upper_case = True
                    new_hashtag = new_hashtag + " " + word
        else:
            enter_in_upper_case = False
            if len(new_hashtag) == 1 and new_hashtag.lower() != "i":
                new_hashtag = ""

            new_hashtag = new_hashtag + " " + word
    return new_hashtag


def read_list_english_words():
    with open("datasets/english_words.txt", encoding="utf8") as f:
        content = f.readlines()
    word_dictionary = [x.strip().lower().rstrip() for x in content]
    word_dictionary = word_dictionary + valid_words
    return word_dictionary


word_dictionary = list(set(read_list_english_words()))
denchant = enchant.Dict("en")


def lemmatization_clean_text(n_samples, column_tolemmatize, completed_path):
    dataset = pd.read_csv(completed_path, sep=",")[:n_samples]

    p = Pool()
    list_data = p.map(preprocess_and_remove_people_in_file, dataset[column_tolemmatize])
    p.close()
    p.join()
    array_result = pd.DataFrame(list_data)

    dataset = pd.concat([dataset, array_result], axis=1)

    return_path = "datasets/filter_merged_text_preprocess.csv"
    dataset.to_csv(return_path, index=False, header=True)
    return return_path
