import re


def re_expressions(list_words):
    if len(list_words) == 0:
        return None
    list_words.sort(reverse=True)
    big_regex_special_names = r'|'.join(map(re.escape, list_words))
    return r"\b(" + big_regex_special_names + r")\b"


def re_expressions_bagofwords(nofake, fake):
    bagofwords = {
        "nofake": re_expressions(nofake),
        "fake": re_expressions(fake),
    }
    return bagofwords


def update_corpus(text, complete_diccionary_by_target):
    list_split = text.split()
    list_window = [" ".join(list_split[i:i + 2]) for i in range(len(list_split) - 1)]
    list_split = list_window
    list_window = [" ".join(list_split[i:i + 3]) for i in range(len(list_split) - 2)]
    list_split = list_split + list_window
    list_window = [" ".join(list_split[i:i + 4]) for i in range(len(list_split) - 3)]
    list_split = list_split + list_window

    for z in list_split:
        if z in complete_diccionary_by_target:
            complete_diccionary_by_target[z] = complete_diccionary_by_target[z] + 1
        else:
            complete_diccionary_by_target[z] = 1
    return complete_diccionary_by_target


def update_dic(text, complete_dic, unique_dic, is_fake_current, number_words, max_value, all_unique_dic, test):
    list_antagonist_emotions = {"fake": "nofake", "nofake": "fake"}
    complete_dic[is_fake_current] = update_corpus(text, complete_dic[is_fake_current])

    for is_fake in ["fake", "nofake"]:
        dataset_to_search_words = complete_dic[list_antagonist_emotions[is_fake]]
        X_tsv_words = complete_dic[is_fake]

        X_tsv_words = {k: v for k, v in sorted(X_tsv_words.items(), reverse=True, key=lambda item: item[1])}

        s = set(dataset_to_search_words.keys())
        main_list = [x for x in X_tsv_words.keys() if x not in s]

        if test:
            list_unique_aux = {}
            for z in main_list[0:800]:
                list_unique_aux[z] = X_tsv_words[z]
            all_unique_dic[is_fake] = list_unique_aux

        unique_dic[is_fake] = []
        for z in main_list[0:number_words]:
            if X_tsv_words[z] > max_value:
                unique_dic[is_fake].append(z)

    bagofwords = re_expressions_bagofwords(unique_dic["nofake"], unique_dic["fake"])
    return complete_dic, unique_dic, bagofwords, all_unique_dic


def search_words_in_bag_of_words(text, bagofwords):
    if bagofwords["nofake"] != None and re.search(bagofwords["nofake"], text, flags=re.IGNORECASE) != None:
        return "nofake"
    elif bagofwords["fake"] != None and re.search(bagofwords["fake"], text, flags=re.IGNORECASE) != None:
        return "fake"
    return None
