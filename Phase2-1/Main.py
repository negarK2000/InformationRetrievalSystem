import pandas as pd
import pickle
import QueryResponse as response
import PositionalResponse as p_response
import Tokenization as tokenizer
import InvertedIndex as indexer
import itertools

file = pd.read_excel(r'Files\IR1_7k_news.xlsx')
data = file.to_dict()
length = len(data['content'])


def read_from_file(path):
    f = open(path, 'rb')
    indexes = pickle.load(f)

    f.close()

    return indexes


def save_to_file(indexes, path):

    with open(path, 'wb') as f:
        pickle.dump(indexes, f, pickle.HIGHEST_PROTOCOL)

    f.close()

    return


def main(k=10):
    positional_index = read_from_file('Files\\positional_inverted_index.pickle')
    tf_df = read_from_file('Files\\tf_df.pickle')
    champion_list = read_from_file('Files\\champion_list.pickle')

    user_query = input("Enter your Query > ")

    query_tokens = tokenizer.get_tokens(user_query)
    query_vector = indexer.get_query_weight(query_tokens)

    print('Result of First Part of Project Second Phase:\n')
    get_tf_idf_response(query_vector, tf_df, positional_index, k)
    print()

    print('Result of First Part of Project Second Phase:\n')
    get_tf_idf_response(query_vector, champion_list, positional_index, k)
    print()

    return


def first_start():
    tokens = tokenizer.process_content(data)
    positional_inverted_index = indexer.get_positional_inverted_index(tokens)

    tf_df = indexer.get_tf_df(positional_inverted_index)
    champion_list = get_champion_list(tf_df)

    save_to_file(positional_inverted_index, 'Files\\positional_inverted_index.pickle')
    save_to_file(tf_df, 'Files\\tf_df.pickle')
    save_to_file(champion_list, 'Files\\champion_list.pickle')
    # save_to_file(tokenizer.training_data, 'Files\\news_training_data.pickle')

    return


def get_champion_list(tf_df, r=30):
    champ = {}

    for term in tf_df.keys():
        term_champ = tf_df[term].copy()
        temp = {-1: term_champ.pop(-1)}

        term_champ = response.sort_dict(term_champ, max)

        temp_champ = dict(itertools.islice(term_champ.items(), r))
        temp.update(temp_champ)
        champ[term] = temp

    return champ


def get_tf_idf_response(query_vector, weight_list, positional_index, k):
    result = response.compute_cosine_scores(query_vector, weight_list, length)

    test(result.copy(), positional_index, query_vector, k)

    for i in range(k):
        if len(result) > 0:
            res_doc = result.popitem()
            print('similarity = {} -> docID = {} : {}'.format(res_doc[1], res_doc[0]+2, data['title'][res_doc[0]]))
            print(data['url'][res_doc[0]])
            print()

        else:
            break

    return


def test(res, positional_index, query, n=10):
    for i in range(n):
        if len(res) > 0:
            res_doc = res.popitem()
            d = res_doc[0]
            print('docID = {}'.format(d+2))

            for term in query.keys():
                if d in positional_index[term].keys():
                    tf = positional_index[term][d][0]
                    print('{} : {}'.format(term, tf))

            print()

        else:
            break

    return


# first_start()
main(5)
