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

id = 0


def process_content(data):
    num = len(data['content'])
    global id
    # iter = 0
    # data['docID'] = {}

    modified_tokens = {}
    for i in range(num):
        modified_tokens[id] = get_tokens(data['content'][i])
        # data['docID'][iter] = id
        id += 1
        # iter += 1

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

    return res
