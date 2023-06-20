import pandas as pd
import pickle
import QueryResponse as response
import Tokenization as tokenizer
import InvertedIndex as indexer
import EmbeddedWord as ew
import Clustering as cluster
import Classification as classify
from gensim.models import Word2Vec as w2v
import numpy as np
from numpy.linalg import norm

pre_prepared_w2v_model = w2v.load('word2vec_model_hazm\\w2v_150k_hazm_300_v2.model')

file0 = pd.read_excel(r'Files\IR1_7k_news.xlsx')
# file0 = pd.read_excel(r'Files\New_IR1_7k_news.xlsx')
data0 = file0.to_dict()

# file1 = pd.read_excel(r'Files\New_IR00_3_11k_News.xlsx')
# file2 = pd.read_excel(r'Files\New_IR00_3_17k_News.xlsx')
# file3 = pd.read_excel(r'Files\New_IR00_3_20k_News.xlsx')

data1 = file1.to_dict()
data2 = file2.to_dict()
data3 = file3.to_dict()

category = ['sport', 'economy', 'health', 'political', 'culture']


def read_from_file(path):
    file = open(path, 'rb')
    indexes = pickle.load(file)

    file.close()

    return indexes


def save_to_file(indexes, path):

    with open(path, 'wb') as file:
        pickle.dump(indexes, file, pickle.HIGHEST_PROTOCOL)

    file.close()

    return


def main(k=10):
    # ....................Phase 2 Data....................
    positional_index = read_from_file('Files\\positional_inverted_index.pickle')
    docs_embedding = read_from_file('Files\\docs_embedding.pickle')

    # ....................Phase 3 Data....................
    new_positional_index = read_from_file('Files\\positional_inverted_index_50k.pickle')
    new_docs_embedding = read_from_file('Files\\docs_embedding_50k.pickle')

    clusters = read_from_file('Files\\docs_clusters_50k.pickle')
    leaders_vector = read_from_file('Files\\clusters_leaders_50k.pickle')

    user_query = input("Enter your Query > ")

    if 'cat' in user_query:
        query_cat = user_query.replace('cat:', '')
        print('Result of Second Part of Project Phase 3:\n< Retrieving Documents with Classification >')

        for c in category:
            if c in query_cat:
                query = query_cat.replace(c, '').strip()

                query_tokens = tokenizer.get_tokens(query)
                query_vector = indexer.get_query_weight(query_tokens)

                get_classified_response(c, query_vector, docs_embedding, positional_index, k)

    query_tokens = tokenizer.get_tokens(user_query)
    query_vector = indexer.get_query_weight(query_tokens)

    print('Result of First Part of Project Phase 3:\n< Retrieving Documents with the help of Clustering >')
    get_clustering_response(query_vector, new_docs_embedding, clusters, leaders_vector, new_positional_index, k)
    print()


def first_start():
    # Data Phase 3
    length = len(data1['id']) + len(data2['id']) + len(data3['id'])

    tokens = tokenizer.process_content(data1)
    tokens1 = tokenizer.process_content(data2)
    tokens2 = tokenizer.process_content(data3)

    tokens.update(tokens1)
    tokens.update(tokens2)

    df1 = pd.DataFrame(data1)
    df1.to_excel("Files\\New_IR00_3_11k_News.xlsx", index_label=['id', 'content', 'topic', 'url', 'docID'])

    df2 = pd.DataFrame(data2)
    df2.to_excel("Files\\New_IR00_3_17k_News.xlsx", index_label=['id', 'content', 'topic', 'url', 'docID'])

    df3 = pd.DataFrame(data3)
    df3.to_excel("Files\\New_IR00_3_20k_News.xlsx", index_label=['id', 'content', 'topic', 'url', 'docID'])

    positional_inverted_index = indexer.get_positional_inverted_index(tokens)

    tf_df = indexer.get_tf_df(positional_inverted_index)
    docs_tf_idf = ew.compute_terms_weight(tf_df, length)

    save_to_file(positional_inverted_index, 'Files\\positional_inverted_index_50k.pickle')
    save_to_file(tf_df, 'Files\\tf_df_50k.pickle')
    save_to_file(docs_tf_idf, 'Files\\docs_tf_idf_50k.pickle')

    docs_embedding = ew.doc_embedding(docs_tf_idf, pre_prepared_w2v_model)
    save_to_file(docs_embedding, 'Files\\docs_embedding_50k.pickle')

    # Clustering
    clusters, leaders_vector = cluster.clustering(docs_embedding)
    save_to_file(clusters, 'Files\\docs_clusters_50k.pickle')
    save_to_file(leaders_vector, 'Files\\clusters_leaders_50k.pickle')

    # Classification
    tokens = read_from_file('Files\\documents_tokens.pickle')
    old_docs_embedding = read_from_file('Files\\docs_embedding.pickle')

    topics = {}
    for i in range(len(old_docs_embedding)):
        doc_vec = old_docs_embedding[i]
        k_nn = classify.get_KNN(tokens[i], doc_vec, docs_embedding, positional_inverted_index)
        topics[i] = classify.classify(k_nn, data1, data2, data3)

    data0['topic'] = topics

    df0 = pd.DataFrame(data0)
    df0.to_excel("Files\\New_IR1_7k_news.xlsx", index_label=['id', 'content', 'url', 'title', 'topic'])


def get_clustering_response(query_vector, new_docs_embedding, clusters, leaders_vector, positional_index, k):
    query_embedding = ew.doc_embedding([query_vector], pre_prepared_w2v_model)

    leaders_score = {}
    for leader in leaders_vector.keys():
        leaders_score[leader] = np.dot(leaders_vector[leader], query_embedding[0]) / (norm(leaders_vector[leader]) * norm(query_embedding[0]))

    sort_scores = response.sort_dict(leaders_score, min)
    query_cluster = clusters[sort_scores.popitem()[0]]

    result = ew.get_similar_docs(query_cluster, query_embedding[0], new_docs_embedding)
    result = response.sort_dict(result, min)

    test(result.copy(), positional_index, query_vector, k)

    len1 = len(data1['id'])
    len2 = len(data2['id'])

    for i in range(k):
        if len(result) > 0:
            res_doc = result.popitem()
            index = res_doc[0]

            if res_doc[0] < len1:
                print_results(data1, res_doc, index)

            elif res_doc[0] < len1 + len2:
                index -= len1
                print_results(data2, res_doc, index)

            else:
                index -= (len1 + len2)
                print_results(data3, res_doc, index)
        else:
            break


def print_results(data, res, index):
    print('similarity = {} -> docID = {} : {}'.format((res[1] + 1) / 2, res[0] + 2, data['url'][index]))
    print()


def get_classified_response(cat, query_vector, positional_index, docs_embedding, k):
    query_embedding = ew.doc_embedding([query_vector], pre_prepared_w2v_model)

    query_docs = []
    for term in query_vector.keys():
        for doc in positional_index[term]:

            if data0['topic'][doc] == cat:
                if doc not in query_docs:
                    query_docs.append(doc)

    query_docs.remove(-1)

    result = ew.get_similar_docs(query_docs, query_embedding[0], docs_embedding)
    result = response.sort_dict(result, min)

    test(result.copy(), positional_index, query_vector, k)

    for i in range(k):
        if len(result) > 0:
            res_doc = result.popitem()
            print('similarity = {} -> docID = {} : {}'.format((res_doc[1] + 1) / 2, res_doc[0] + 2, data0['title'][res_doc[0]]))
            print(data0['url'][res_doc[0]])
            print()

        else:
            break


def get_embedded_word_response(query_vector, docs_embedding, positional_index, k):
    query_embedding = ew.doc_embedding([query_vector], pre_prepared_w2v_model)

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
            print('similarity = {} -> docID = {} : {}'.format((res_doc[1]+1)/2, res_doc[0]+2, data0['title'][res_doc[0]]))
            print(data0['url'][res_doc[0]])
            print()

        else:
            break


def test(res, positional_index, query, n=10):
    for i in range(n):
        if len(res) > 0:
            res_doc = res.popitem()
            id = res_doc[0]
            print('docID = {}'.format(id+2))

            for term in query.keys():
                if id in positional_index[term].keys():
                    tf = positional_index[term][id][0]
                    print('{} : {}'.format(term, tf))

            print()

        else:
            break

    return


first_start()
main(3)
