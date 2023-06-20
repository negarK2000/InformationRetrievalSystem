import pandas as pd
import Tokenization as tokenizer
import InvertedIndex as indexer
import Query
import pickle
import Graph as g


# 'content' - 'url' - 'title'
file = pd.read_excel (r'Files\IR1_7k_news.xlsx')
data = file.to_dict()


def first_start():

    # First Part of Project - tokenizing
    tokens = tokenizer.process_content(data['content'])

    # Second Part of Project - indexing
    inverted_index = indexer.get_positional_inverted_index(tokens)

    save_to_file(inverted_index, 'Files\indexes.pickle')

    return inverted_index


def save_to_file(indexes, path):

    with open(path, 'wb') as file:
        pickle.dump(indexes, file, pickle.HIGHEST_PROTOCOL)

    file.close()

    return


def read_from_file(path):
    indexes = {}

    file = open(path, 'rb')
    indexes = pickle.load(file)

    file.close()

    return indexes


# getting the tokens
# inverted_index = first_start()
inverted_index = read_from_file('Files\\indexes.pickle')


def get_query():
    # Third Part of Project - search query
    user_query = input("Enter your Query > ")
    query_tokens = tokenizer.get_tokens(user_query)
    result = Query.respond_to_the_query(query_tokens, inverted_index)

    # Forth Part of Project - answer query
    # Related Rank = The higher the rank, the result is more related to the Query
    n = len(result) - 1
    printed = []
    for i in range(n + 1):
        docId_list = result[n - i]

        if n > 0:
            print('The Related Rank:    {}'.format(n - i + 1))

            for docId in docId_list:
                if docId not in printed:
                    printed.append(docId)
                    print(data['title'][docId])

            print()

        else:
            for docId in docId_list:
                print(data['title'][docId])

    return


def cal_zipf_law():
    freq = tokenizer.words_frequency

    freq = sort_list(freq)

    stopwords = tokenizer.stopwords

    new_freq = {}
    for word in freq:
        if word not in stopwords:
            new_freq[word] = freq[word]

    save_to_file(freq, 'Files\\freq_with_stopwords.pickle')
    save_to_file(new_freq, 'Files\\freq_without_stopwords.pickle')

    return freq, new_freq


def zipf_law():
    # for the first time
    # freq1, freq2 = cal_zipf_law()

    freq1 = read_from_file('Files\\freq_with_stopwords.pickle')
    freq2 = read_from_file('Files\\freq_without_stopwords.pickle')

    # before removing the stop words - most frequent words
    g.plot_zipf_law(freq1, "Checking Zipf's Law Before Removing the Stop words")

    # after removing the stop words - most frequent words
    g.plot_zipf_law(freq2, "Checking Zipf's Law After Removing the Stop words")
    return


def sort_list(words):
    words_list = list(words.items())

    l = len(words_list)
    for i in range(l - 1):
        for j in range(i + 1, l):
            if words_list[i][1] < words_list[j][1]:
                temp = words_list[i]
                words_list[i] = words_list[j]
                words_list[j] = temp

    return dict(words_list)


def heap_law():
    # for the first time
    # heap_dict = tokenizer.heap_dict
    # stem_heap_dict = tokenizer.heap_dict
    #
    # save_to_file(heap_dict, 'Files\\heap_law_tokens.pickle')
    # save_to_file(stem_heap_dict, 'Files\\heap_law_stem_tokens.pickle')

    heap_dict = read_from_file('Files\\heap_law_tokens.pickle')
    stem_heap_dict = read_from_file('Files\\heap_law_stem_tokens.pickle')

    g.plot_heap_law(heap_dict, 'Checking Heap Law without Stem tokens')
    g.plot_heap_law(stem_heap_dict, 'Checking Heap Law with Stem tokens')

    return

# heap_law()
# zipf_law()

get_query()
