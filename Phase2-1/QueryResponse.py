import math as m


def compute_cosine_scores(q_weight, tf_df, length):
    scores = [0] * length
    len_docs = [0] * length
    res = {}

    for term in q_weight:

        if term in tf_df.keys():
            documents = tf_df[term]

            for docId in list(documents.keys())[1:]:
                temp = tf_df[term][docId] * m.log10(length/tf_df[term][-1])
                scores[docId] += q_weight[term] * temp
                len_docs[docId] += temp ** 2

    for i in range(length):
        temp = scores[i]
        if temp > 0:
            res[i] = temp / m.sqrt(len_docs[i])

    return sort_dict(res, min)


def sort_dict(res_dict, exp):
    res_list = list(res_dict.items())
    l = len(res_list)

    for i in range(l - 1):
        for j in range(i + 1, l):
            if exp(res_list[i][1] , res_list[j][1]) == res_list[j][1]:
                t = res_list[i]
                res_list[i] = res_list[j]
                res_list[j] = t

    return dict(res_list)
