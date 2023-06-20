import EmbeddedWord as ew
import QueryResponse as response


# KNN Algorithm
def classify(knn, data1, data2, data3, k=15):
    category_num = {'sport': 0, 'economy': 0, 'health': 0, 'political': 0, 'culture': 0}
    len1 = len(data1['id'])
    len2 = len(data2['id'])

    for i in range(k):
        index = knn[i]

        if knn[i] < len1:
            n_class = data1['topic'][index]

        elif knn[i] < len1 + len2:
            index -= len1
            n_class = data2['topic'][index]

        else:
            index -= (len2 + len1)
            n_class = data3['topic'][index]

        for cat in category_num.keys():
            if n_class == cat:
                category_num[cat] += 1

    max = 0
    res_cat = ''
    for c in category_num.keys():
        if max < category_num[c]:
            max = category_num[c]
            res_cat = c

    return res_cat


def get_KNN(tokens, test_vector, learned_vectors, positional_index, k=15):

    related_docs = []
    for term in tokens:
        try:
            for doc in positional_index[term]:
                if doc not in related_docs:
                    related_docs.append(doc)
        except KeyError:
            continue

    related_docs.remove(-1)

    result = ew.get_similar_docs(related_docs, test_vector, learned_vectors)
    result = response.sort_dict(result, min)

    knn = []
    for i in range(k):
        if len(result) > 0:
            res_doc = result.popitem()
            knn.append(res_doc[0])
        else:
            break

    return knn
