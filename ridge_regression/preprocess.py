import os
import pickle
import json
import argparse
import numpy as np
import re
import pandas as pd
from  build_vocab import Vocabulary
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_cap = TfidfVectorizer(input='filename')
vectorizer_tag = TfidfVectorizer(input='filename')

def preprocess_train(path):
    img_path = os.path.join(path, 'features_train/')
    cap_path = os.path.join(path, 'descriptions_train/')
    tag_path = os.path.join(path, 'tags_train/')
    # print("----- Loading Vocab -----")
    # vocab_cap = pickle.load(open('vocab_cap.pkl', 'rb'))
    # vocab_tag = pickle.load(open('vocab_tag.pkl', 'rb'))

    img_filename = img_path + 'features_resnet1000intermediate_train.csv'
    images_df = pd.read_csv(img_filename, header=None)
    images = images_df.values
    img_ids, img_features = images[:, 0], images[:, 1:]
    print(img_ids.shape)
    print(img_features.shape)
    print(img_features[0, 0])
    print(type(img_features[0, 0]))

    cap_filenames, tag_filenames = [], []


    output = {'image_id':[], 'image_2048':[], 'captions':[], 'tags':[]}
    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        tag_filename = tag_path + filename +'.txt'

        output['image_id'].append(id)
        output['image_2048'].append(img_f)

        cap_filenames.append(cap_filename)
        tag_filenames.append(tag_filename)

    weighted_captions = vectorizer_cap.fit_transform(cap_filenames)
    output['captions'] = weighted_captions.toarray()
    print(weighted_captions.shape)

    weighted_tags = vectorizer_tag.fit_transform(tag_filenames)
    output['tags'] = weighted_tags.toarray()
    print(weighted_tags.shape)
    # print(vectorizer_tag.get_feature_names())


    output['image_2048'] = np.array(output['image_2048']).astype(float)


    length = len(output['image_id'])
    with open('train.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)


def preprocess_test(path):
    img_path = os.path.join(path, 'features_test/')
    cap_path = os.path.join(path, 'descriptions_test/')
    tag_path = os.path.join(path, 'tags_test/')


    img_filename = img_path + 'features_resnet1000intermediate_test.csv'
    images_df = pd.read_csv(img_filename, header=None)
    images = images_df.values
    img_ids, img_features = images[:, 0], images[:, 1:]

    output = {'image_id':[], 'image_2048':[], 'captions':[], 'tags':[]}
    captions = []
    cap_filenames, tag_filenames = [], []

    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        tag_filename = tag_path + filename +'.txt'

        output['image_id'].append(id)
        output['image_2048'].append(img_f)

        # captions.append(" ".join(cap))
        cap_filenames.append(cap_filename)
        tag_filenames.append(tag_filename)

    weighted_captions = vectorizer_cap.transform(cap_filenames)
    output['captions'] = weighted_captions.toarray()
    print(weighted_captions.shape)

    weighted_tags = vectorizer_tag.transform(tag_filenames)
    output['tags'] = weighted_tags.toarray()
    print(weighted_tags.shape)
    # print(vectorizer_tag.get_feature_names())

    output['image_2048'] = np.array(output['image_2048']).astype(float)


    length = len(output['image_id'])
    with open('test.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)


if __name__ == '__main__':
    data_path = '../data'
    preprocess_train(data_path)
    preprocess_test(data_path)

