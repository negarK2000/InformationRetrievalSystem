def respond_to_the_query(query_tokens, inverted_index):
    if len(query_tokens) == 0:
        return []

    else:
        temp = phrase_query(query_tokens, inverted_index)

        if len(query_tokens) == 1:
            return [[docId for docId in inverted_index[query_tokens[0]].keys() if docId != -1]]

        res = rank_results(temp)

    return res


# res = {word: {docId: [pos, ...]}, ...}, ...}
def phrase_query(tokens, indexes):
    res = {}

    for token in tokens:
        temp = {}
        if token in indexes.keys():
            temp = indexes[token]

        if len(temp) > 1:
            res[token] = temp

    return res


# input = two words = {docId: [pos, ...]}, ...}
# output = list of two words  intersect doc Id = [docId, ...]
def positional_intersect(word1, word2, dist):
    res = []

    iter_word1 = iter(word1)
    iter_word2 = iter(word2)

    next(iter_word1)
    next(iter_word2)

    docId1 = next(iter_word1)
    docId2 = next(iter_word2)

    while True:
        try:

            if docId1 == docId2:
                l = []

                for pos1 in word1[docId1][1:]:
                    for pos2 in word2[docId2][1:]:
                        if abs(pos1 - pos2) <= dist:
                            l.append(pos2)

                        elif pos2 > pos1:
                            break

                    while len(l) != 0 and abs(l[0] - pos1) > dist:
                        l.pop(0)

                    if len(l) != 0:
                        res.append(docId1)

                docId1 = next(iter_word1)
                docId2 = next(iter_word2)

            elif docId1 < docId2:
                docId1 = next(iter_word1)

            else:
                docId2 = next(iter_word2)

        except StopIteration:
            break

    return res


def get_joint(list1, list2):
    res = []

    if len(list1) == 0 or len(list2) == 0:
        return []

    iter_list1 = iter(list1)
    iter_list2 = iter(list2)

    id1 = next(iter_list1)
    id2 = next(iter_list2)

    while True:
        try:

            if id1 == id2:
                res.append(id1)

                id1 = next(iter_list1)
                id2 = next(iter_list2)

            elif id1 < id2:
                id1 = next(iter_list1)

            else:
                id2 = next(iter_list2)

        except StopIteration:
            break

    return res


# input = {word: {docId: [pos, ...]}, ...}, ...}
# res = [[docId_two_word], [docId_three_word], ..., [docId_all_word]]
def rank_results(result):
    res = []
    l = len(result)

    res.append([])
    for word in result.keys():
        res[0].extend(docId for docId in result[word] if docId != -1 and docId not in res[0])

    word_lists = []
    two_words = []
    for i in range(l):
        iter1 = list(result.keys())[i]
        for j in range(i+1,l):
            iter2 = list(result.keys())[j]
            temp = positional_intersect(result[iter1], result[iter2], j-i)

            word_lists.append(temp)
            two_words.extend([docId for docId in temp if docId not in two_words])

    res.append(two_words)

    while len(word_lists) > 1:
        res.append([])
        temp_list = []

        n = len(word_lists)
        for i in range(0, n, 2):

            try:
                temp = get_joint(word_lists[i], word_lists[i+1])
                res[-1].extend(id for id in temp if id not in res[-1])

                temp_list.append(temp)

            except IndexError:
                temp_list.append(word_lists[i])

        word_lists = temp_list

    return res
