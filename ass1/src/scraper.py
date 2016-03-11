from pytrends.pyGTrends import pyGTrends

import numpy as np
import getpass

USER = ''
PASS = ''
DATA_DIR = '../data/'

def scrape():
    # load queries
    hpv_queries = load_file('hpv_queries.txt').split('\n')
    mfr_queries = load_file('mfr_queries.txt').split('\n')

    trends = pyGTrends
    # connect to google
    trends = pyGTrends(USER, PASS)

    # scrape hpv
    hpv_matrix = []
    for query in hpv_queries:
        trends.request_report(keywords=query, date='01/2011 60m', geo='DK')
        raw = trends.get_data()
        hpv_matrix.append(parse_counts(raw))
    hpv_matrix = np.array(hpv_matrix)
    hpv_matrix = np.transpose(hpv_matrix)
    np.savetxt(DATA_DIR + 'hpv_data.txt', hpv_matrix, fmt='%i', delimiter=",")
    # scrape mfr
    mfr_matrix = []
    for query in mfr_queries:
        trends.request_report(keywords=query, date='01/2011 60m', geo='DK')
        raw = trends.get_data()
        mfr_matrix.append(parse_counts(raw))
    mfr_matrix = np.array(mfr_matrix)
    mfr_matrix = np.transpose(mfr_matrix)
    np.savetxt(DATA_DIR + 'mfr_data.txt', mfr_matrix, fmt='%i', delimiter=",")


# --------- help functions --------------

def parse_counts(raw):
    results = {}
    # add all months
    for year in range(11,16):
        for month in range(1,13):
            if month<10:
                str_month = '0' + str(month)
            else:
                str_month = str(month)
            results['20' + str(year) + '-' + str_month] = 0

    for line in raw.split('\n'):
        if line.startswith('201') and line[4] == '-':
            month = line[5:7]
            results[line[:7]] += int(line.split(',')[1])
    return [item for _,item in sorted(results.items())]

def load_file(filename):
    with open(DATA_DIR + filename) as f:
        output = f.read()
    f.close()
    return output


# MAIN
if __name__ == '__main__':
    if USER == '' or PASS == '':
        print('Google account is necessarry for downloading data from Google Trends, please enter your username and password:')
        USER = input('Username: ')
        PASS = getpass.getpass()
    scrape()
