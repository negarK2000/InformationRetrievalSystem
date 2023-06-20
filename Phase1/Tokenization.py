import hazm as tp
import re

stopwords_file = open('Files\Persian-StopWords.txt', mode='r', encoding='utf-8')

stopwords = []
sw = stopwords_file.readline()
while sw:
    stopwords.append(sw.strip())
    sw = stopwords_file.readline()

stopwords_file.close()

exceptions_file = open('Files\exceptions.txt', mode='r', encoding='utf-8')

exceptions = []
exc = exceptions_file.readline()
while exc:
    exceptions.append(exc.strip())
    exc = exceptions_file.readline()

exceptions_file.close()

words_frequency = {}
heap_num = [1, 500, 1000, 1500, 2000]
all_tokens = [] # T
all_stem_tokens = [] # T
all_stem_diff_tokens = [] # M
all_diff_tokens = [] # M
heap_dict = {}
stem_heap_dict = {}


def process_content(data):
    num = len(data)

    heap = True
    modified_tokens = {}
    for i in range(num):
        modified_tokens[i] = get_tokens(data[i])

        # if i in heap_num:
        #     heap_dict[len(all_tokens)] = len(all_diff_tokens)
        #     stem_heap_dict[len(all_stem_tokens)] = len(all_stem_diff_tokens)
        #
        #     if i == heap_num[-1]:
        #         heap = False

    return modified_tokens


def get_tokens(text):
    text = re.sub(r'(https.+)', '', text)

    text = re.sub('[\W_]+',' ', text)

    text = re.sub(r'[0-9]','', text)

    tokens = tp.word_tokenize(text)

    filtered_tokens = [word for word in tokens if word not in stopwords]

    res = []
    for token in filtered_tokens:
        if token not in exceptions:

            stem = tp.Stemmer().stem(token)

            if stem != '':
                res.append(stem)
        else:
            res.append(token)

    # cal_the_freq(tokens)

    # if heap:
    #     cal_the_token_num(tokens, res)

    return res


def cal_the_freq(tokens):

    for token in tokens:
        if token not in words_frequency:
            words_frequency[token] = 1

        else:
            words_frequency[token] += 1

    return


def cal_the_token_num(tokens, stems):

    all_tokens.extend(tokens)
    all_stem_tokens.extend(stems)

    all_diff_tokens.extend(token for token in tokens if token not in all_diff_tokens)
    all_stem_diff_tokens.extend(token for token in stems if token not in all_diff_tokens)

    return
