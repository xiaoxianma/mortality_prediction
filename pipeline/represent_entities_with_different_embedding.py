import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
import glove
from glove import Corpus

import collections
import gc

import warnings

warnings.filterwarnings("ignore")

data_path = "../data/intermediate/"

new_notes = pd.read_pickle(f"{data_path}ner_df.p")  # med7
w2vec = Word2Vec.load("embeddings/word2vec.model")
fasttext = FastText.load("embeddings/fasttext.model")


null_index_list = []
for i in new_notes.itertuples():

    if len(i.ner) == 0:
        null_index_list.append(i.Index)
new_notes.drop(null_index_list, inplace=True)

med7_ner_data = {}

for ii in new_notes.itertuples():

    p_id = ii.SUBJECT_ID
    ind = ii.Index

    try:
        new_ner = new_notes.loc[ind].ner
    except:
        new_ner = []

    unique = set()
    new_temp = []

    for j in new_ner:
        for k in j:

            unique.add(k[0])
            new_temp.append(k)

    if p_id in med7_ner_data:
        for i in new_temp:
            med7_ner_data[p_id].append(i)
    else:
        med7_ner_data[p_id] = new_temp

pd.to_pickle(med7_ner_data, f"{data_path}/new_ner_word_dict.pkl")


def mean(a):
    return sum(a) / len(a)


data_types = [med7_ner_data]
data_names = ["new_ner"]

for data, names in zip(data_types, data_names):
    new_word2vec = {}
    print("w2vec starting..")
    for k, v in data.items():

        patient_temp = []
        for i in v:
            try:
                patient_temp.append(w2vec[i[0]])
            except:
                avg = []
                num = 0
                temp = []

                if len(i[0].split(" ")) > 1:
                    for each_word in i[0].split(" "):
                        try:
                            temp = w2vec[each_word]
                            avg.append(temp)
                            num += 1
                        except:
                            pass
                    if num == 0:
                        continue
                    avg = np.asarray(avg)
                    t = np.asarray(map(mean, zip(*avg)))
                    patient_temp.append(t)
        if len(patient_temp) == 0:
            continue
        new_word2vec[k] = patient_temp

    #############################################################################
    print("fasttext starting..")

    new_fasttextvec = {}

    for k, v in data.items():

        patient_temp = []

        for i in v:
            try:
                patient_temp.append(fasttext[i[0]])
            except:
                pass
        if len(patient_temp) == 0:
            continue
        new_fasttextvec[k] = patient_temp

    #############################################################################

    print("combined starting..")
    new_concatvec = {}

    for k, v in data.items():
        patient_temp = []
        #     if k != 6: continue
        for i in v:
            w2vec_temp = []
            try:
                w2vec_temp = w2vec[i[0]]
            except:
                avg = []
                num = 0
                temp = []

                if len(i[0].split(" ")) > 1:
                    for each_word in i[0].split(" "):
                        try:
                            temp = w2vec[each_word]
                            avg.append(temp)
                            num += 1
                        except:
                            pass
                    if num == 0:
                        w2vec_temp = [0] * 100
                    else:
                        avg = np.asarray(avg)
                        w2vec_temp = np.asarray(map(mean, zip(*avg)))
                else:
                    w2vec_temp = [0] * 100

            fasttemp = fasttext[i[0]]

            appended = np.append(fasttemp, w2vec_temp, 0)
            patient_temp.append(appended)
        if len(patient_temp) == 0:
            continue
        new_concatvec[k] = patient_temp

    print(len(new_word2vec), len(new_fasttextvec), len(new_concatvec))
    pd.to_pickle(new_word2vec, data_path + names + "_word2vec_dict.pkl")
    pd.to_pickle(new_fasttextvec, data_path + names + "_fasttext_dict.pkl")
    pd.to_pickle(new_concatvec, data_path + names + "_combined_dict.pkl")


diff = set(new_fasttextvec.keys()).difference(set(new_word2vec))
for i in diff:
    del new_fasttextvec[i]
    del new_concatvec[i]
print(len(new_word2vec), len(new_fasttextvec), len(new_concatvec))


pd.to_pickle(new_word2vec, data_path + "new_ner" + "_word2vec_limited_dict.pkl")
pd.to_pickle(new_fasttextvec, data_path + "new_ner" + "_fasttext_limited_dict.pkl")
pd.to_pickle(new_concatvec, data_path + "new_ner" + "_combined_limited_dict.pkl")
