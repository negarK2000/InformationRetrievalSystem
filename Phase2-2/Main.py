import pandas as pd
import pickle
import QueryResponse as response
import Tokenization as tokenizer
import InvertedIndex as indexer
import EmbeddedWord as ew
from gensim.models import Word2Vec as w2v

file = pd.read_excel(r'Files\IR1_7k_news.xlsx')
data = file.to_dict()
length = len(data['content'])

pre_prepared_w2v_model = w2v.load('word2vec_model_hazm\\w2v_150k_hazm_300_v2.model')
news_w2v_model = w2v.load('Files\\news_word2vec_model.pickle')


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

    docs_embedding1 = read_from_file('Files\\docs_embedding1.pickle')
    docs_embedding2 = read_from_file('Files\\docs_embedding2.pickle')

    user_query = input("Enter your Query > ")

    query_tokens = tokenizer.get_tokens(user_query)
    query_vector = indexer.get_query_weight(query_tokens)

    print('Result of Second Part of Project Second Phase:')
    print('*Based on the news Embedded Words*\n')
    get_embedded_word_response(news_w2v_model, query_vector, docs_embedding1, positional_index, k)
    print()

    print('Result of Second Part of Project Second Phase:')
    print('*Based on the pre-prepared Embedded Words*\n')
    get_embedded_word_response(pre_prepared_w2v_model, query_vector, docs_embedding2, positional_index, k)
    print()

    return


def first_start():
    tokens = tokenizer.process_content(data)
    positional_inverted_index = indexer.get_positional_inverted_index(tokens)

    tf_df = indexer.get_tf_df(positional_inverted_index)

    save_to_file(positional_inverted_index, 'Files\\positional_inverted_index.pickle')
    save_to_file(tf_df, 'Files\\tf_df.pickle')
    # save_to_file(tokenizer.training_data, 'Files\\news_training_data.pickle')

    ew.train_word2vec_model(tokenizer.training_data)

    docs_tf_idf = ew.compute_terms_weight(tf_df, length)
    # save_to_file(docs_tf_idf, 'Files\\docs_tf_idf.pickle')

    docs_embedding1 = ew.doc_embedding(docs_tf_idf, news_w2v_model)
    docs_embedding2 = ew.doc_embedding(docs_tf_idf, pre_prepared_w2v_model)

    save_to_file(docs_embedding1, 'Files\\docs_embedding1.pickle')
    save_to_file(docs_embedding2, 'Files\\docs_embedding2.pickle')

    return


def get_embedded_word_response(w2v_model, query_vector, docs_embedding, positional_index, k):
    query_embedding = ew.doc_embedding([query_vector], w2v_model)

    query_docs = []
    for term in query_vector.keys():
        for doc in positional_index[term]:
            if doc not in query_docs:
                query_docs.append(doc)

    query_docs.remove(-1)

    result = ew.get_similar_docs(query_docs, query_embedding[0], docs_embedding)
    result = response.sort_dict(result, min)

    test(result.copy(), positional_index, query_vector, k)

    for i in range(k):
        if len(result) > 0:
            res_doc = result.popitem()
            print('similarity = {} -> docID = {} : {}'.format((res_doc[1]+1)/2, res_doc[0]+2, data['title'][res_doc[0]]))
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
