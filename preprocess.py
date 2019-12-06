import os
import pickle
import json
import argparse
import numpy as np
import re
import pandas as pd

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
    
    output = []
    for i, (id_p, img_f) in enumerate(zip(img_ids, img_features)):
        data = {}
        id = id_p.split('/')[1]
        filename = id[:-4]
        cap_filename = cap_path + filename +'.txt'
        tag_filename = tag_path + filename +'.txt'

        data['image_id'] = id
        data['image_2048'] = img_f
        data['captions'] = open(cap_filename, 'r').read().splitlines()
        data['tags'] = open(tag_filename, 'r').read().splitlines()
        
        output.append(data)

    length = len(output)
    train, val = output[:int(length*0.8)], output[int(length*0.8):]
    print(len(train))
    print(len(val))
    with open('train.pkl', 'wb') as outfile:
        pickle.dump(train, outfile)
    with open('val.pkl', 'wb') as outfile:
        pickle.dump(val, outfile)

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
        data['captions'] = open(cap_filename, 'r').read().splitlines()
        data['tags'] = open(tag_filename, 'r').read().splitlines()
        
        output.append(data)

    # print(output)
    print(images.shape)
    with open('test.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)


if __name__ == '__main__':
    data_path = './data'
    preprocess_train(data_path)
    preprocess_test(data_path)
