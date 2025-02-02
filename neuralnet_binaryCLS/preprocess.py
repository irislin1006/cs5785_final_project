import os
import pickle
import json
import argparse
import numpy as np
import re
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")

def preprocess_train(path):
    img_path = os.path.join(path, 'features_train/')
    cap_path = os.path.join(path, 'descriptions_train/')
    tag_path = os.path.join(path, 'tags_train/')

    img_filename = img_path + 'features_resnet1000intermediate_train.csv'
    images_df = pd.read_csv(img_filename, header=None)
    images = images_df.values
    img_ids, img_features = images[:, 0], images[:, 1:]
    print(img_ids.shape)
    print(img_features.shape)
    print(img_features[0, 0])
    print(type(img_features[0, 0]))

    captions = []
    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        captions.append(" ".join(open(cap_filename, 'r').read().splitlines()))

    vectorizer.fit(captions)

    output = []
    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        data = {}
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        tag_filename = tag_path + filename +'.txt'

        data['image_id'] = id
        data['image_2048'] = img_f

        cap = " ".join(open(cap_filename, 'r').read().splitlines())
        data['captions'] = vectorizer.transform([cap])[0]
        data['tags'] = open(tag_filename, 'r').read().splitlines()

        output.append(data)

    length = len(output)
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(output, test_size=0.2, random_state=22)
    print(len(train))
    print(len(val))
    with open('train.pkl', 'wb') as outfile:
        pickle.dump(train, outfile)
    with open('val.pkl', 'wb') as outfile:
        pickle.dump(val, outfile)
    print("tfidf len", len(data['captions']))
    print(data['captions'])

def preprocess_test(path):
    img_path = os.path.join(path, 'features_test/')
    cap_path = os.path.join(path, 'descriptions_test/')
    tag_path = os.path.join(path, 'tags_test/')

    img_filename = img_path + 'features_resnet1000intermediate_test.csv'
    images_df = pd.read_csv(img_filename, header=None)
    images = images_df.values
    img_ids, img_features = images[:, 0], images[:, 1:]

    output = []
    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        # print(img_f.shape)
        data = {}
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        tag_filename = tag_path + filename +'.txt'

        data['image_id'] = id
        data['image_2048'] = img_f
        cap = " ".join(open(cap_filename, 'r').read().splitlines())
        data['captions'] = vectorizer.transform([cap])[0]
        data['tags'] = open(tag_filename, 'r').read().splitlines()

        output.append(data)

    # print(output)
    print(images.shape)
    with open('test.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)


if __name__ == '__main__':
    data_path = '../data'
    preprocess_train(data_path)
    preprocess_test(data_path)
