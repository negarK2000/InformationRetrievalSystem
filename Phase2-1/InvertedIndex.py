import math as m


# input = {docId: {word: [pos, ...], ... }}
# res = {word: {docId: [pos, ...]}, ...}, ...}
# docId = -1 => value = total Number of words in documents
def get_positional_inverted_index(tokens_dict):
    positional_index = index_documents(tokens_dict)

    inverted_index = {}
    for docId in positional_index.keys():
        for word in positional_index[docId].keys():

            if word in inverted_index.keys():

                if docId in inverted_index[word].keys():
                    inverted_index[word][docId].extend(positional_index[docId][word][1:])
                else:
                    inverted_index[word][docId] = positional_index[docId][word]

                inverted_index[word][-1] += positional_index[docId][word][0]

            else:
                inverted_index[word] = {}
                inverted_index[word][-1] = positional_index[docId][word][0]
                inverted_index[word][docId] = positional_index[docId][word]

    return inverted_index


# input = [word, ...]
# output = {word: [wordNum, pos, ...], ...}
def index_positions(tokens):
    file_index = {}
    for index, word in enumerate(tokens):
        if word in file_index.keys():
            file_index[word].append(index)
            file_index[word][0] += 1
        else:
            file_index[word] = [1]
            file_index[word].append(index)

    return file_index


# input = {docId: [word, ...], ...}
# res = {docId: {word: [pos, ...]}, ...}
def index_documents(tokens_dict):
    res = {}
    for docId in tokens_dict.keys():
        res[docId] = index_positions(tokens_dict[docId])

    return res


# input = {term: {docId: [pos, ...]}, ...}, ...}
# res = {term: {-1: df(term), docId: log10(tf(t,d))+1, ...}, ...}
# docId = -1 => value = df(word)
def get_tf_df(terms_dict):
    res = {}

    for term in terms_dict.keys():
        if term not in res.keys():
            res[term] = {}
            res[term][-1] = 0

        for doc in list(terms_dict[term].keys())[1:]:
            res[term][doc] = m.log10(terms_dict[term][doc][0]) + 1
            res[term][-1] += 1

    return res


# input = [term, ...]
# res = {term: log10(tf(t,q))+1, ...}
def get_query_weight(terms):
    res = {}

    for term in terms:
        if term not in res.keys():
            res[term] = 0

        res[term] += 1

    for term in res:
        freq = res[term]
        res[term] = m.log10(freq) + 1

    return res
