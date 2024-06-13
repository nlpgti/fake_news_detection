import datetime
import os
import warnings
from datetime import date
from itertools import accumulate

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def cum_concat(x):
    list_return = list(accumulate(x))
    return list_return


def method_compare_avg(dataset_per_user, feature_name):
    dataset_per_user[feature_name + '_avg'] = dataset_per_user[feature_name].cumsum()
    dataset_per_user[feature_name + '_avg'] = dataset_per_user[feature_name + '_avg'] / dataset_per_user['tweet_num']
    dataset_per_user[feature_name + '_higher'] = np.where(
        dataset_per_user[feature_name + '_avg'] > dataset_per_user[feature_name], False, True)
    return dataset_per_user


def lambda_funtions(x):
    if x['text_list_acc'].count(x["text"]) > 1:
        x['text_duplicated'] = True
    else:
        x['text_duplicated'] = False
    x['year_week'] = str(date.fromtimestamp(x["timestamp"]).year) + str(
        date.fromtimestamp(x["timestamp"]).isocalendar()[1])
    x["account_age"] = (datetime.datetime.fromtimestamp(x["timestamp"]) - datetime.datetime.fromtimestamp(
        x["user_registration"])).days
    return x


def aggregate_user_per_day(user):
    dataset_per_user = dataset[dataset["user_name"] == user]
    dataset_per_user = dataset_per_user.sort_values(by=['user_name', 'timestamp'])
    dataset_per_user.reset_index(inplace=True)

    dataset_per_user['tweet_num'] = range(1, 1 + len(dataset_per_user))

    list_features_to_avg = ["char_counts", "links_count", "links_repeated_count", "user_favourite_count",
                            "user_friends_count"
        , "user_followers_count", "bad_words_count", "retweet_count", "difficult_words_count",
                            "user_ratio_friends_followers"
        , 'pos_prop_PRON', 'pos_prop_AUX', 'pos_prop_DET', 'pos_prop_ADJ', 'pos_prop_NOUN', 'pos_prop_PUNCT',
                            'word_count', 'tweet_favorite_count', 'hashtags_count', 'uppercase_words_count']
    for z in list_features_to_avg:
        dataset_per_user = method_compare_avg(dataset_per_user, z)

    dataset_per_user = dataset_per_user.apply(lambda x: lambda_funtions(x), axis=1)

    index_duplicated_yearweek = pd.Index(dataset_per_user["year_week"]).duplicated()
    dataset_per_user['week_num'] = (~index_duplicated_yearweek).astype(int)
    dataset_per_user['week_num'] = dataset_per_user["week_num"].cumsum()
    dataset_per_user['tweet_freq_week'] = dataset_per_user['tweet_num'] / dataset_per_user['week_num']

    dataset_per_user.pop("text_list")
    dataset_per_user.pop("text_list_acc")
    return dataset_per_user


def incremental_text_list():
    global dataset
    dataset = dataset.sort_values(by=["timestamp"], ascending=True)
    dataset = dataset.reset_index(drop=True)
    dataset["text_preprocessed"].fillna("", inplace=True)
    dataset.fillna(0, inplace=True)

    dataset["union"] = 1
    dataset["text_list"] = dataset["text"].apply(lambda x: [x])

    f = lambda x: cum_concat([i for i in x])
    b = dataset.groupby(['union'])['text_list'].apply(f)
    list_aux = [item for sublist in b for item in sublist]
    dataset["text_list_acc"] = list_aux

dataset = "READ DATASET AND RUN incremental_text_list() and aggregate_user_per_day()"
