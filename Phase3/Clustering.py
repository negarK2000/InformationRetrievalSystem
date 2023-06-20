from numpy import random
import QueryResponse as response
import numpy as np
from numpy.linalg import norm

# clusters = {leaderID: [followerID, ...], ...}
clusters = {}
# leaders_vector = {leaderID: leader_vector, ...}
leaders_vector = {}
pre_leaders = {}

VECTOR_SIZE = 300


# K-means Algorithm
def clustering(docs_vectors, k=10):
    rand_list = []
    length = len(docs_vectors)
    n = length // (k-1)

    i = 0
    while len(rand_list) < k:
        x = random.randint(length)
        if x not in rand_list:
            rand_list.append(x)
            clusters[i] = []
            leaders_vector[i] = docs_vectors[x]
            i += 1

    chose_followers(docs_vectors.copy(), n)

    while True:
        pre_leaders = leaders_vector.copy()
        chose_leaders()

        chose_followers(docs_vectors.copy(), n)

        if has_converge(pre_leaders):
            break

    return clusters, leaders_vector


def chose_leaders():

    for leaderId in clusters.keys():
        sum_followers = np.zeros(VECTOR_SIZE)

        for follower in clusters[leaderId]:
            sum_followers += follower

        leaders_vector[leaderId] = sum_followers / len(clusters[leaderId])


def chose_followers(docs_vectors, n):

    for leaderId in leaders_vector.keys():
        doc_scores = {}
        num = len(docs_vectors)
        clusters[leaderId] = []

        for docId in range(num):
            doc_scores[docId] = np.dot(docs_vectors[docId], leaders_vector[leaderId]) / (norm(docs_vectors[docId]) * norm(leaders_vector[leaderId]))

        result = response.sort_dict(doc_scores, min)

        rest_docs = {}
        for i in range(num):
            if len(result) > 0:
                res_doc = result.popitem()

                if i < n:
                    clusters[leaderId].append(res_doc[0])

                else:
                    rest_docs[res_doc[0]] = docs_vectors[res_doc[0]]

            else:
                break

        docs_vectors = rest_docs


def has_converge(pre_leaders, diff = 0.1):
    CONVERGED = True
    num = len(leaders_vector)
    list_diff = [diff] * VECTOR_SIZE
    difference = np.array(list_diff)

    for i in range(num):
        new = np.array(leaders_vector[i])
        pre = np.array(pre_leaders[i])
        res = abs(np.subtract(new, pre)) > difference
        if res.all():
            CONVERGED = False
            break

    return CONVERGED