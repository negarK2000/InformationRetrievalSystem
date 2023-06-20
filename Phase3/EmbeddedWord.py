from gensim.models import Word2Vec as w2v
import multiprocessing
import numpy as np
from numpy.linalg import norm
import math as m

VECTOR_SIZE=300


def train_word2vec_model(training_data):
    cores = multiprocessing.cpu_count()
    print('Number of all docs in training data = ', len(training_data))
    print('Number of all tokens in training data = ', sum([len(x) for x in training_data]))

    w2v_model = w2v(min_count=3, window=4, vector_size=VECTOR_SIZE, alpha=0.04, workers=cores - 1, sample=1e-5, sg= 1, hs= 0)

    w2v_model.build_vocab(training_data)
    w2v_model_vocab_size = len(w2v_model.wv)
    print('vocab size = ', w2v_model_vocab_size)

    w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=30)

    w2v_model.save('Files\\news_word2vec_model.pickle')
    return


def compute_terms_weight(tf_df, length):
    docs_tf_idf = [0] * length

    for term in tf_df.keys():
        for docId in list(tf_df[term].keys())[1:]:

            if docs_tf_idf[docId] == 0:
                docs_tf_idf[docId] = {}

            docs_tf_idf[docId][term] = tf_df[term][docId] * m.log10(length / tf_df[term][-1])

    return docs_tf_idf


def doc_embedding(docs_tf_idf, w2v_model):
    docs_embedding = {}

    for doc in docs_tf_idf:
        if type(doc) != type({}):
            continue

        doc_vec = np.zeros(VECTOR_SIZE)
        weight_sum = 0

        for token, weight in doc.items():
            try:
                doc_vec += w2v_model.wv[token] * weight
                weight_sum += weight

            except KeyError:
                continue

        if weight_sum != 0:
            docs_embedding[docs_tf_idf.index(doc)] = doc_vec / weight_sum

    return docs_embedding


def get_similar_docs(query_docs, query_embedding, docs_embedding):
    doc_scores = {}

    for doc in query_docs:
        try:
            doc_scores[doc] = np.dot(docs_embedding[doc], query_embedding) / (norm(docs_embedding[doc]) * norm(query_embedding))
        except KeyError:
            continue

    return doc_scores
