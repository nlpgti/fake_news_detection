import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
from ast import literal_eval
from utils.constants import filtered_bad_words_ORES


def count_url(x):
    urls = []
    matches = re.finditer(
        r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        x["text"], re.MULTILINE | re.IGNORECASE)
    for matchNum, match in enumerate(matches, start=1):
        urls.append(match.group())
    x["links_count"] = len(urls)
    x["links_list"] = urls
    x["links_repeated_count"] = len(urls) - len(set(urls))
    return x


def has_image_video(x):
    if ".png" in x["links_list"] or ".jpg" in x["links_list"] or ".svg" in x["links_list"] or ".mp4" in x["links_list"]:
        x["has_image_video"] = True
    else:
        x["has_image_video"] = False
    return x


def filter_bad_words(x):
    list_lemmas = x.split()
    bad = 0
    regex = r'|'.join(filtered_bad_words_ORES)
    regex = r"\b(" + regex + r")\b"
    for w in list_lemmas:
        bad = bad + len(re.findall(regex, w))
    return bad


def depth(x):
    if type(x) is dict and x:
        return 1 + max(depth(x[a]) for a in x)
    return 0


def first_level_depth(x):
    if type(x[list(x.keys())[0]]) is dict:
        return len(x[list(x.keys())[0]].keys())
    else:
        return 0


def add_all_process_features(x):
    x["user_ratio_friends_followers"] = x['user_friends_count'] / x['user_followers_count']
    x["char_counts"] = len(x["text"])
    x["hashtags_count"] = len(x["hashtags"])
    x["bad_words_count"] = filter_bad_words(x["text_preprocessed"])
    x["uppercase_words_count"] = sum(map(str.isupper, x["text"].split()))

    x = count_url(x)
    x = has_image_video(x)

    x["depth_retweets"] = depth(x["related_ids"])
    x["first_level_retweets"] = first_level_depth(x["related_ids"])

    return x


def no_lambda_functions(dataset):
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy[["user_background_image", "user_profile_image"]]
    dataset["user_has_profile_image"] = dataset_copy.any(axis='columns')

    dataset['user_description'] = dataset['user_description'].fillna("")
    dataset['user_has_description'] = np.where(dataset['user_description'] != "", True, False)


def add_features(path):
    annotation_data_info = pd.read_csv(path)
    annotation_data_info["hashtags"] = annotation_data_info["hashtags"].apply(literal_eval)
    annotation_data_info["related_ids"] = annotation_data_info["related_ids"].apply(literal_eval)

    no_lambda_functions(annotation_data_info)
    annotation_data_info = annotation_data_info.apply(lambda x: add_all_process_features(x), axis=1)

    annotation_data_info.to_csv("datasets/filter_merged_text_preprocess_features_process.csv", index=False, header=True)

    return "datasets/filter_merged_text_preprocess_features_process.csv"
