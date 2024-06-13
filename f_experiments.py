import os
import warnings

from e_ML_river import experiment_run
from e_ML_river import prepare_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def only_params(X, y, X_tsv,list_params_eval,list_model_hyper_params):
    for model_hyper_params in  list_model_hyper_params:
        one_element = {'X': X, 'y': y, "X_tsv": X_tsv, 'max_df_in': 0.7, 'min_df_in': 0.01, 'ngram_range_in': (1, 3),
                       "model_hyper_params": model_hyper_params, "model_type": model_hyper_params["model_name"],
                       'list_params_eval': list_params_eval, "number_words": 700, "max_value": 1,
                       "only_classifier": False,"dictionary": False,
                       "balanced_train": False, "only_params_eval": True, "verbose": False}

        result=experiment_run(one_element)
        print(result["latex"])



def params_ngrams(X, y, X_tsv,list_params_eval,list_model_hyper_params):
    for model_hyper_params in list_model_hyper_params:
        one_element = {'X': X, 'y': y, "X_tsv": X_tsv, 'max_df_in': 0.7, 'min_df_in': 0.01, 'ngram_range_in': (1, 3),
                       "model_hyper_params": model_hyper_params, "model_type": model_hyper_params["model_name"],
                       'list_params_eval': list_params_eval, "number_words": 700, "max_value": 1,
                       "only_classifier": False, "dictionary": False,
                       "balanced_train": False, "only_params_eval": False, "verbose": False}

        result=experiment_run(one_element)
        print(result["latex"])

def params_ngrams_dic(X, y, X_tsv,list_params_eval,list_model_hyper_params):

    for model_hyper_params in list_model_hyper_params:
        one_element = {'X': X, 'y': y, "X_tsv": X_tsv, 'max_df_in': 0.7, 'min_df_in': 0.01, 'ngram_range_in': (1, 3),
                       "model_hyper_params": model_hyper_params, "model_type": model_hyper_params["model_name"],
                       'list_params_eval': list_params_eval, "number_words": 700, "max_value": 1,
                       "only_classifier": False,  "dictionary": True,
                       "balanced_train": False, "only_params_eval": False, "verbose": False}

        result=experiment_run(one_element)
        print(result["latex"])


if __name__ == '__main__':
    X, y, X_tsv = prepare_dataset("datasets/filter_merged_text_preprocess_features_process_incremental.csv")

    list_params_eval = ['account_age', 'user_verified',
    'user_time_zone', 'user_protected', 'user_has_profile_image', 'user_has_description',
    'user_followers_count', 'user_friends_count', 'user_favourite_count','difficult_words_count','char_counts','links_count',
                        'links_repeated_count','bad_words_count','retweet_count',
                        "user_ratio_friends_followers",
                        'pos_prop_PRON', 'pos_prop_AUX', 'pos_prop_DET', 'pos_prop_ADJ', 'pos_prop_NOUN','pos_prop_PUNCT',
                        'word_count', 'tweet_favorite_count', 'hashtags_count', 'uppercase_words_count',


                        'user_favourite_count_avg', 'user_favourite_count_higher',
                        'user_friends_count_avg', 'user_friends_count_higher',
                        'user_followers_count_avg', 'user_followers_count_higher',
                        'difficult_words_count_avg', "difficult_words_count_higher",
                        'char_counts_avg','char_counts_higher',
                        'links_count_avg', 'links_count_higher',
                        'links_repeated_count_avg', 'links_repeated_count_higher',
                        'bad_words_count_avg', 'bad_words_count_higher',
                        'retweet_count_avg', 'retweet_count_higher',
                        "user_ratio_friends_followers_avg","user_ratio_friends_followers_higher",
                        'pos_prop_PRON_avg','pos_prop_PRON_higher',
                        'pos_prop_AUX_avg','pos_prop_AUX_higher',
                        'pos_prop_DET_avg','pos_prop_DET_higher',
                        'pos_prop_ADJ_avg','pos_prop_ADJ_higher',
                        'pos_prop_NOUN_avg','pos_prop_NOUN_higher',
                        'pos_prop_PUNCT_avg','pos_prop_PUNCT_higher',
                        'word_count_avg','word_count_higher',
                        'tweet_favorite_count_avg','tweet_favorite_count_higher',
                        'hashtags_count_avg','hashtags_count_higher',
                        'uppercase_words_count_avg','uppercase_words_count_higher',

    'text_duplicated','polarity', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise',
    'flesch_reading_ease', 'mcalpine_eflaw', 'reading_time', 'has_image_video', 'retweeted',
    'tweet_favorited',  'depth_retweets','first_level_retweets', 'tweet_freq_week']



    list_model_hyper_params = []
    # list_model_hyper_params.append({'model_name':"arfc",'n_clusters': 10, 'n_models': 200, 'max_features': 50, 'lambda_value': 50})
    # list_model_hyper_params.append({'model_name':"hatc",'n_clusters': 10, 'max_depth': 50, 'tie_threshold': 0.05, 'max_size': 200})
    # list_model_hyper_params.append({'model_name':"htc",'n_clusters': 10, 'max_depth': 50, 'tie_threshold': 0.05, 'max_size': 50})
    list_model_hyper_params.append({'model_name':"gnb",'n_clusters': 10})

    print("***************** only_params *************")
    only_params(X, y, X_tsv,list_params_eval,list_model_hyper_params)
    print("\n***************** params_ngrams *************")
    params_ngrams(X, y, X_tsv, list_params_eval,list_model_hyper_params)
    print("\n***************** params_ngrams_dic *************")
    params_ngrams_dic(X, y, X_tsv, list_params_eval,list_model_hyper_params)







