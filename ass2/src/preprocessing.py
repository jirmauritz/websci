from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import codecs
import csv

DATA_DIR = '../data/'
RAW = 'data.csv'
LEX = 'sentiment_lexicon/sentiment_lexicon.tff'
U_MAT = 'first/unigrams.txt'
B_MAT = 'first/bigrams.txt'
REVIEWS = 'second/reviews.txt'
LABELS = 'first/labels.txt'

FEATURES = 200

def preprocess():
    # load
    with open(DATA_DIR + RAW, 'r') as f:
        raw = csv.reader(f)
        raw = list(raw)[1:] # first line are labels
        f.close()
    raw = list(map(lambda x: [x[i] for i in [1,3]], raw)) # take second and forth column (sentiment and text)
    # delete non evaluated reviews
    raw = [x for x in raw if x[0] == '1' or x[0] == '0' or x[0] == '-1']
    # chop to labels and data
    labels = [int(row[0]) for row in raw]
    rews = [row[1] for row in raw]
    # load sentiment lexicon
    lex = loadLex()
    # unigrams
    uni_mat = grams(rews, lex, 1, 1)
    # bigrams
    bi_mat = grams(rews, lex, 1, 2)
    # lex sentiment
    lex_sent = lex_sentiment(rews, lex)
    uni_mat['lex'] = lex_sent
    bi_mat['lex'] = lex_sent

    # save matrices
    np.savetxt(DATA_DIR + U_MAT, uni_mat.as_matrix(), fmt='%i')
    np.savetxt(DATA_DIR + B_MAT, bi_mat.as_matrix(), fmt='%i')
    np.savetxt(DATA_DIR + LABELS, labels, fmt='%i')
    # save reviews
    f = open(DATA_DIR + REVIEWS, 'w')
    f.write('\nSTOPWORD\n'.join(rews))
    f.close()


def lex_sentiment(rews, lex):
    sentiments = []
    for rew in rews:
        sent = 0
        for word in map(lambda x: removePunctuation(x).lower(), rew.split()):
            if word in lex:
                sent += lex[word]
        sentiments.append(sent)
    return sentiments


def grams(rews, lex, min_gram, max_gram):
    # get dataframe
    vectorizer = CountVectorizer(ngram_range=(min_gram, max_gram), binary=1, max_features=FEATURES)
    mat = vectorizer.fit_transform(rews)
    df = pd.DataFrame(mat.toarray(), columns = vectorizer.get_feature_names())
    return df

def sentiment(gram, lex):
    # neg, _, _, _, ... --> negative
    # pos, 0, 0, ... --> positive
    # 0, 0 --> neutral
    lex_labels = map(lambda x: lex[x] if x in lex else 0, gram.split())
    if -1 in lex_labels:
        return -1
    elif 1 in lex_labels:
        return 1
    else:
        return 0

def loadLex():
    lex = {}
    with open(DATA_DIR + LEX, 'r') as f:
        for line in f.readlines():
            # parse word
            word = line.split()[2][6:]
            sentiment = line.split()[5][14:]
            if sentiment == 'positive':
                lex[word] = 1
            elif sentiment == 'negative':
                lex[word] = -1
        f.close()
    return lex


def removePunctuation(text):
    return ''.join(list(map (_spacify, text)))

    
# -------- private ---------

def _spacify(char):
    if char.isalpha():
        return char
    return ' '


# MAIN
preprocess()

