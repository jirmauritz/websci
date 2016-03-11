import string

# constants
DATA_DIR = '../data/'
ALLOWED_CHARS = [' ', 'Æ', 'Ø', 'Å', 'æ', 'ø', 'å']

def preprocess():
    # load
    hpv = loadDesc('HPV_desc')
    mfr = loadDesc('MFR_desc')
    # remove punctuation
    hpv = (removePunctuation(hpv[0]), removePunctuation(hpv[1]))
    mfr = (removePunctuation(mfr[0]), removePunctuation(mfr[1]))
    # lower
    hpv = (hpv[0].lower(), hpv[1].lower())
    mfr = (mfr[0].lower(), mfr[1].lower())
    # bagify
    hpv = (hpv[0].split(), hpv[1].split())
    mfr = (mfr[0].split(), mfr[1].split())
    # unique words
    hpv = (set(hpv[0]), set(hpv[1]))
    mfr = (set(mfr[0]), set(mfr[1]))
    # remove stop words
    stop_words = load_stop_words('stopwords.txt')
    hpv = (remove_stop_words(hpv[0], stop_words), remove_stop_words(hpv[1], stop_words))
    mfr = (remove_stop_words(mfr[0], stop_words), remove_stop_words(mfr[1], stop_words))
    # filter only those, who are in the both sets
    hpv = both(hpv[0], hpv[1])
    mfr = both(mfr[0], mfr[1])
    # save
    save_queries(hpv, 'hpv_queries.txt')
    save_queries(mfr, 'mfr_queries.txt')


# ---------- help functions ------------

def loadDesc(filename):
    # load files
    first = _load_file('web/' + filename + '1.txt')
    second = _load_file('web/' + filename + '2.txt')
    return (first, second)

def removePunctuation(text):
    return ''.join(list(map (_spacify, text)))

def load_stop_words(filename):
    raw = _load_file(filename)
    lines = raw.split('\n')
    return [line.split(' ')[0] for line in lines]

def remove_stop_words(text, stop_words):
    return list(filter((lambda x: x not in stop_words), text))

def both(set1, set2):
    final = set()
    for word in set1:
        if word in set2:
            final.add(word)
    return final

def save_queries(data_set, filename):
    f = open(DATA_DIR + filename, 'w')
    f.write('\n'.join(data_set))
    f.close()

    
# -------- private ---------

def _spacify(char):
    if char.isalnum() or char in ALLOWED_CHARS:
        return char
    return ' '

def _load_file(filename):
    with open(DATA_DIR + filename) as f:
        output = f.read()
    f.close()
    return output

if __name__ == '__main__':
    preprocess()
