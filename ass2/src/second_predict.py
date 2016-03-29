import numpy as np
import scipy.stats.mstats as mstats
from sklearn import svm as sk_svm, neural_network
from sklearn.cross_validation import KFold
from pathlib import Path
import os
from datetime import datetime

DATA_DIR = 'data/'
U_MAT = 'first/unigrams.txt'
B_MAT = 'first/bigrams.txt'
LABELS = 'first/labels.txt'
REVIEWS = 'second/reviews.txt'
OUTPUT = 'second/output.txt'
RESULT = 'stanford_res.txt'

def predict():
    # load labels
    labels = np.loadtxt(DATA_DIR + LABELS, dtype=(int))
    # baseline - always predicts majority
    baseline(labels)

    # annotate with the Stanford Sentiment Analyzator
    # !!!!! WARNING !!!!! this could take several hours
    annotate()

    # predict sentiment from output of the SSA
    predict = process_output()

    # save predictions
    np.savetxt(DATA_DIR + RESULT, np.array(predict), fmt='%i')
    
    print('method: DEEP_LEARNING, ACC: %.3f' %
            (accuracy(predict, labels[0:len(predict)])))

def baseline(labels):
    """
    Baseline for testing, predict majority value for any input.
    Usually, the majority is positive feedback.
    """
    mode = mstats.mode(labels).mode[0]
    print('method: BASELINE, ACC: %.3f' %
            (accuracy([mode]*len(labels), labels)))

    print()

def annotate():
    touch_file(DATA_DIR + OUTPUT)
    # call stanford sentiment analysis, output store to file
    now = datetime.now()
    os.system('java -cp "stanford-nlp/*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file ' +  
        DATA_DIR + REVIEWS + ' >> ' + DATA_DIR + OUTPUT)
    then = datetime.now()
    print('time: ' +  str((then - now).seconds) + ' sec')

def process_output():
    # process output
    with open(DATA_DIR + OUTPUT) as f:
        predict = []
        feel = set()
        for line in f:
            if line == '  Positive\n':
                feel.add('pos')
            if line == '  Negative\n':
                feel.add('neg')
            if line == 'STOPWORD\n':
                predict.append(positive_approach(feel))
                feel = set()
        predict.append(positive_approach(feel))
        f.close()
    return predict

def neutral_approach(feel):
    if 'pos' in feel and 'neg' in feel:
        return 0
    elif 'pos' in feel:
        return 1
    elif 'neg' in feel:
        return -1
    else:
        return 0

def positive_approach(feel):
    if 'pos' in feel:
        return 1
    elif 'neg' in feel:
        return -1
    else:
        return 0




# --------- help functions --------------

def touch_file(filename):
    out_file = Path(filename)
    if out_file.exists():
        out_file.unlink()
    out_file.touch()


def accuracy(labels1, labels2):
    s = 0
    for i in range(len(labels1)):
        s += labels1[i] == labels2[i]
    return s/len(labels1)



#MAIN
predict()
