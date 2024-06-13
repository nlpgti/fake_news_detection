import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import datetime
import json
import pandas as pd


def get_timestamp(x):
    return datetime.datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y').timestamp()


def prepare_data_set_with_additional_info(list_json_tweet, name):
    data = pd.DataFrame.from_records(list_json_tweet)
    data["source"] = name
    data["timestamp"] = data["created_at"].apply(lambda x: get_timestamp(x))
    data["user_registration"] = data["user_registration"].apply(
        lambda x: get_timestamp(x))
    return data


def get_info_by_tweet(file_dir, related_ids, fake):
    with open(file_dir) as f:
        d = json.load(f)
        entities = d["entities"]
        d["user_mentions"] = entities["user_mentions"]
        d["hashtags"] = entities["hashtags"]

        d["tweet_favorited"] = d["favorited"]
        d["tweet_favorite_count"] = d["favorite_count"]

        user = d["user"]
        d["user_background_image"] = user["profile_use_background_image"]
        d["user_profile_image"] = user["default_profile_image"]
        d["user_verified"] = user["verified"]
        d["user_followers_count"] = user["followers_count"]
        d["user_description"] = user["description"]
        d["user_friends_count"] = user["friends_count"]
        d["user_location"] = user["location"]
        d["user_favourite_count"] = user["favourites_count"]
        d["user_name"] = user["name"]
        d["user_registration"] = user["created_at"]
        d["user_time_zone"] = user["time_zone"]
        d["user_protected"] = user["protected"]

        d["related_ids"] = related_ids
        d["is_fake"] = fake

    return d


def create_dataset(path, name):
    list_dirs = [x for x in os.walk(path) if any(word in x[0] for word in ['source-tweets'])]
    list_json_tweet = []
    for z in list_dirs:
        file_dir = z[0].replace("source-tweets", "annotation.json")
        with open(file_dir) as f:
            d = json.load(f)
            fake = d["is_rumour"]

        file_dir = z[0].replace("source-tweets", "") + "/" + "structure.json"
        with open(file_dir) as f:
            d = json.load(f)
            related_ids = d

        file_dir = z[0] + "/" + z[2][0]

        d = get_info_by_tweet(file_dir, related_ids, fake)
        list_json_tweet.append(d)

    annotation_data_info = prepare_data_set_with_additional_info(list_json_tweet, name)
    annotation_data_info.to_csv("datasets/" + name + ".csv", index=False, header=True)


def filter_dataset(path, name):
    dataset = pd.read_csv(path + name + ".csv")
    dataset = dataset[
        ["source", "id", "timestamp", "text", "tweet_favorite_count", "retweeted", "user_mentions", "hashtags",
         "retweet_count", "tweet_favorited"
            , "user_background_image", "user_profile_image", "user_verified", "user_followers_count",
         "user_description",
         "user_friends_count",
         "user_favourite_count", "user_name", "user_registration", "user_time_zone", "user_protected", "related_ids",
         "is_fake"]]
    dataset.to_csv(path + "filter_" + name + ".csv", index=False, header=True)
    return path + "filter_" + name + ".csv"


def merge_dataset(list, path, name):
    list_dataframes = []
    for z in list:
        list_dataframes.append(pd.read_csv(z))
    dataset_combined = pd.concat(list_dataframes)

    dataset_combined = dataset_combined.loc[dataset_combined['is_fake'].isin(["nonrumour", "rumour"])]
    str_to_bool = {'nonrumour': "nofake", 'rumour': "fake"}
    dataset_combined['is_fake'] = dataset_combined['is_fake'].map(str_to_bool)

    dataset_combined = dataset_combined.drop_duplicates(subset='id', keep="first")
    dataset_combined.to_csv(path + name + ".csv", index=False, header=True)

