#coding=utf-8
import re
import nltk
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
stops1 = set(stopwords.words("spanish"))

def clean_sent(sent):
    sent = sent.lower()
    sent = re.sub(u'[_"\-;%()|+&=*%.,!?:#$@\[\]/]',' ',sent)
    sent = re.sub('¡',' ',sent)
    sent = re.sub('¿',' ',sent)
    sent = re.sub('Á','á',sent)
    sent = re.sub('Ó','ó',sent)
    sent = re.sub('Ú','ú',sent)
    sent = re.sub('É','é',sent)
    sent = re.sub('Í','í',sent)
    return sent
def cleanSpanish(df):
    df['spanish1'] = df.spanish1.map(lambda x: ' '.join([ word for word in
                                                         nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
    df['spanish2'] = df.spanish2.map(lambda x: ' '.join([ word for word in
                                                         nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
def removeSpanishStopWords(df, stop):
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
                                                         if word not in stop]).encode('utf-8'))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
                                                         if word not in stop]).encode('utf-8'))

def data_preprocessing():

    # Training data
    df_train_en_sp = pd.read_csv('./input/cikm_english_train_20180516.txt', sep='	', header=None,
                                 error_bad_lines=False)
    df_train_sp_en = pd.read_csv('./input/cikm_spanish_train_20180516.txt', sep='	', header=None,
                                 error_bad_lines=False)
    df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
    df_train_sp_en.columns = ['spanish1', 'english1', 'spanish2', 'english2', 'result']
    train1 = pd.DataFrame(pd.concat([df_train_en_sp['spanish1'], df_train_sp_en['spanish1']], axis=0))
    train2 = pd.DataFrame(pd.concat([df_train_en_sp['spanish2'], df_train_sp_en['spanish2']], axis=0))
    train_data = pd.concat([train1, train2], axis=1).reset_index()
    train_data = train_data.drop(['index'], axis=1)
    result = pd.DataFrame(pd.concat([df_train_en_sp['result'], df_train_sp_en['result']], axis=0)).reset_index()
    result = result.drop(['index'], axis=1)
    # pd.get_dummies(result['result']).head()
    train_data['result'] = result

    # Evaluation data
    test_data = pd.read_csv('./input/cikm_test_a_20180516.txt', sep='	', header=None, error_bad_lines=False)
    test_data.columns = ['spanish1', 'spanish2']


    cleanSpanish(train_data)
    removeSpanishStopWords(train_data, stops1)
    cleanSpanish(test_data)
    removeSpanishStopWords(test_data, stops1)

    train_data.replace('', np.nan, inplace=True)
    dirty_data = train_data[train_data.isnull().any(axis=1)]
    print 'dirty sample count:', dirty_data.shape[0]
    print 'positive dirty training sample:', len(dirty_data[dirty_data['result'] == 1])
    print 'negative dirty training sample:', len(dirty_data[dirty_data['result'] == 0])

    train_data = train_data.dropna()
    test_data.replace('', np.nan, inplace=True)
    test_data = test_data.dropna()
    print 'Train sample count:', train_data.shape[0], 'Test sample count:', test_data.shape[0]

    train_data.columns = ['s1', 's2', 'label']
    test_data.columns = ['s1', 's2']

    train_data.to_csv("input/cleaned_train.csv", index=False)
    test_data.to_csv("input/cleaned_test.csv", index=False)

def get_embedding(word_dict, embedding_path, embedding_dim=300):
    # find existing word embeddings
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))

    # handling unknown embeddings
    for word in word_dict:
        if word not in word_vec:
            # If word not in word_vec, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding
    print "Filled missing words' embeddings."
    print "Embedding Matrix Size: ", len(word_vec)

    return word_vec

def save_embed(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print 'Embedding saved'

def load_embed(path):
    with open(path, 'rb') as f:
        return pickle.load(f)