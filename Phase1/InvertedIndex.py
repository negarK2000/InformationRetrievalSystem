# input = {docId: {word: [pos, ...], ... }}
# res = {word: {docId: [pos, ...]}, ...}, ...}
# docId = -1 => value = total Number of words in a document
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
            num = file_index[word][0]
            file_index[word][0] = num + 1
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
