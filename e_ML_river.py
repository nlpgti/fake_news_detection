import os
import time
import warnings

import pandas as pd
import scipy
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion

from utils.create_dictionary import update_dic, search_words_in_bag_of_words
from utils.utils import print_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from river import ensemble, feature_selection
from river import stream
from river import cluster


def normalize_params(X_train, list_params_eval, verbose):
    X_train_counts_aux = pd.DataFrame()

    for param in list_params_eval:
        X_train[param] = X_train[param] * 1
        X_train_counts_aux[param] = X_train[param]

    if verbose:
        print(X_train_counts_aux.shape)
    return X_train_counts_aux


def add_params(X_train_counts, X_train, list_params_eval, list_n_grams, to_dataframe):
    if len(list_params_eval) > 1:
        X_train_counts_aux = pd.DataFrame()

        for param in list_params_eval:
            X_train[param] = X_train[param] * 1

            X_train_counts_aux[param] = X_train[param]

        if to_dataframe:
            X_train_counts = pd.DataFrame(X_train_counts.toarray(), columns=list_n_grams)
            X_train_counts = X_train_counts.join(X_train_counts_aux)

        else:
            X_train_counts_aux = scipy.sparse.csr_matrix(X_train_counts_aux.values)
            X_train_counts = hstack((X_train_counts, X_train_counts_aux), format='csr')
    else:
        if to_dataframe:
            X_train_counts = pd.DataFrame(X_train_counts.toarray(), columns=list_n_grams)

    return (X_train_counts)


def add_params_and_get_column_names(X_train_counts, X_train, list_params_eval, list_n_grams):
    X_train_counts = add_params(X_train_counts, X_train, list_params_eval, list_n_grams, to_dataframe=True)
    list_n_grams = [*list_n_grams, *list_params_eval]
    return (X_train_counts, list_n_grams)


def n_grams(X, max_df_in, min_df_in, ngram_range_in, max_features_in, verbose, all):
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        min_df=min_df_in,
        max_df=max_df_in,
        ngram_range=ngram_range_in,
        max_features=max_features_in,
    )
    result_w = count_vect_word.fit_transform(X)

    if all:
        count_vect_char = CountVectorizer(
            analyzer='char',
            lowercase=True,
            min_df=min_df_in,
            max_df=max_df_in,
            ngram_range=ngram_range_in,
            max_features=max_features_in,
        )
        result_c = count_vect_char.fit_transform(X)

        count_vect_char_wb = CountVectorizer(
            analyzer='char_wb',
            lowercase=True,
            min_df=min_df_in,
            max_df=max_df_in,
            ngram_range=ngram_range_in,
            max_features=max_features_in,
        )
        result_cwb = count_vect_char_wb.fit_transform(X)

        if (verbose):
            print(str(result_w.shape)
                  + " " + str(result_c.shape) + " " + str(result_cwb.shape))
        X_train_counts = FeatureUnion([("CountVectorizerWords", count_vect_word),
                                       ("CountVectorizerChars", count_vect_char),
                                       ("CountVectorizerCharsWB", count_vect_char_wb)
                                       ]).transform(X)

        list_n_grams = count_vect_word.get_feature_names_out() + count_vect_char.get_feature_names_out() + count_vect_char_wb.get_feature_names_out()
        return (X_train_counts, list_n_grams)

    else:
        if (verbose):
            print(str(result_w.shape))
        X_train_counts = FeatureUnion([("CountVectorizerWords", count_vect_word)
                                       ]).transform(X)

        list_n_grams = count_vect_word.get_feature_names_out()
        return (X_train_counts, list_n_grams)


def prepare_dataset(path):
    X_tsv = pd.read_csv(path, sep=",")

    X_tsv = X_tsv.sort_values(by=["timestamp"], ascending=True)
    X_tsv = X_tsv.reset_index(drop=True)

    X_tsv["text_preprocessed"].fillna("", inplace=True)
    X_tsv["user_time_zone"].fillna("", inplace=True)
    X_tsv.fillna(0, inplace=True)
    y = X_tsv["is_fake"]

    labels, uniques = pd.factorize(X_tsv['user_time_zone'])
    X_tsv['user_time_zone'] = labels
    X = X_tsv["text_preprocessed"]

    return X, y, X_tsv


def experiment_run(data):
    n_data_set = None

    init_value = time.time()
    model_hyper_params = data["model_hyper_params"]
    n_clusters = model_hyper_params["n_clusters"]
    model_name = data["model_type"]
    list_params_eval = data["list_params_eval"]
    only_classifier = data["only_classifier"]
    dictionary_enable = data["dictionary"]
    balanced_train = data["balanced_train"]

    verbose = data["verbose"]

    y = data["y"][:n_data_set]
    X_tsv = data["X_tsv"][:n_data_set]

    if data["only_params_eval"]:
        X = normalize_params(X_tsv, list_params_eval, verbose)
    else:
        X = data["X"][:n_data_set]
        X, list_n_grams = n_grams(X, max_df_in=data["max_df_in"], min_df_in=data["min_df_in"],
                                  ngram_range_in=data["ngram_range_in"], max_features_in=None, verbose=verbose,
                                  all=False)
        X, list_n_grams = add_params_and_get_column_names(X, X_tsv, list_params_eval, list_n_grams)

    selector = feature_selection.VarianceThreshold()

    cluster_model = cluster.KMeans(n_clusters=n_clusters, seed=1)

    list_models = []
    for z in range(n_clusters):
        model_to_analyse = None
        if model_name == "arfc":
            model_to_analyse = ensemble.AdaptiveRandomForestClassifier(n_models=model_hyper_params["n_models"],
                                                                       max_features=model_hyper_params["max_features"],
                                                                       lambda_value=model_hyper_params["lambda_value"],
                                                                       seed=1)
        elif model_name == "hatc":
            model_to_analyse = HoeffdingAdaptiveTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                               tie_threshold=model_hyper_params["tie_threshold"],
                                                               max_size=model_hyper_params["max_size"])
        elif model_name == "htc":
            model_to_analyse = HoeffdingTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                       tie_threshold=model_hyper_params["tie_threshold"],
                                                       max_size=model_hyper_params["max_size"])
        elif model_name == "gnb":
            model_to_analyse = GaussianNB()
        list_models.append(
            {"model": model_to_analyse,
             "elements": 0})

    list_y_pred = []
    list_y = []

    list_text_preprocessed = []
    list_all_unique_dic = []
    list_y_pred_dic = []
    list_y_dic = []

    words_freq_dic = {"fake": {}, "nofake": {}}
    unique_words_freq_dic = {"fake": {}, "nofake": {}}
    unique_selected_words_dic = {"fake": [], "nofake": []}

    unique_words_regular_expressions = None

    index = 0
    count_y_fake = 0
    count_y_nofake = 0

    X_train = X
    y_train = y

    for x_river, y_river in stream.iter_pandas(X_train, y_train):
        if selector is not None:
            x_river = selector.learn_one(x_river).transform_one(x_river)

        dictionary_prediction = None

        if dictionary_enable:
            if index > 321:
                dictionary_prediction = search_words_in_bag_of_words(X_tsv["text_preprocessed"].iloc[index],
                                                                     unique_words_regular_expressions)
                if dictionary_prediction != None:
                    list_y_pred_dic.append(dictionary_prediction)
                    list_y_dic.append(y_river)
            words_freq_dic, unique_selected_words_dic, unique_words_regular_expressions, unique_words_freq_dic = update_dic(
                X_tsv["text_preprocessed"].iloc[index],
                words_freq_dic, unique_selected_words_dic
                , y_river, data["number_words"],
                data["max_value"], unique_words_freq_dic, only_classifier)

        if only_classifier:
            dictionary_prediction = None

        cluster_num = cluster_model.predict_one(x_river)
        rf = list_models[cluster_num]
        if dictionary_prediction == None:
            y_pred = rf["model"].predict_one(x_river)
            if rf["elements"] > 0:
                if y_pred != None:
                    list_y_pred.append(y_pred)
                    list_y.append(y_river)

                if only_classifier:
                    list_text_preprocessed.append(X_tsv["text_preprocessed"].iloc[index])
                    list_all_unique_dic.append(unique_words_freq_dic.copy())
        else:
            list_y_pred.append(dictionary_prediction)
            list_y.append(y_river)

        istrain = False
        if count_y_fake >= count_y_nofake and y_river == "nofake":
            count_y_nofake = count_y_nofake + 1
            istrain = True
        elif y_river == "fake":
            count_y_fake = count_y_fake + 1
            istrain = True

        if istrain or not balanced_train:
            cluster_model = cluster_model.learn_one(x_river)

            try:
                rf["model"] = rf["model"].learn_one(x_river, y_river)
                rf["elements"] = rf["elements"] + 1
            except  Exception as e:
                print(e)
        if verbose:
            print(index)
        index = index + 1

    data["X_tsv"] = data["y"] = data["X"] = ""

    print_result = print_metrics(model_name, data, list_y, list_y_pred, time.time() - init_value)

    pd_predicions = pd.DataFrame(columns=["list_y", "list_y_pred"])
    pd_predicions["list_y"] = list_y
    pd_predicions["list_y_pred"] = list_y_pred
    pd_predicions.to_csv("datasets/predictions_analyse.csv", index=False, header=True)

    if verbose:
        print(time.time() - init_value)

    return print_result
