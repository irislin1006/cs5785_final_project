import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from build_vocab import Vocabulary
import Constants
import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os,sys
from nltk.tokenize import TweetTokenizer
nlp = TweetTokenizer()

class ImageSearchDataset(Dataset):
    def __init__(self, vocab_cap, vocab_tag, data, feature=None, period="24"):
        self.data = data
        self.vocab_cap = vocab_cap
        self.vocab_tag = vocab_tag
        self.max_len = 20

    def __getitem__(self, index):
        # 0: recipe_id, 1:instruction/recipe, 2:list of ingres
        image_id = self.data[index]['image_id']
        img_features = self.data[index]['image_2048']
        tags = self.data[index]['tags']

        # tokenize tags
        tag = " ".join(tags).replace(":", " ").split(" ")
        tag_bow = [0]*len(self.vocab_tag)
        for t in tag:
            tag_bow[self.vocab_tag(t)] = 1
        return image_id, img_features, tag_bow

    def collate_fn(self, data):
        image_id, img_fea, tags_bow = zip(*data)

        #image_id = torch.Tensor(image_id)
        img_fea = torch.FloatTensor(img_fea)
        tags = torch.FloatTensor(tags_bow)
        return image_id, (img_fea, tags)

    def __len__(self):
        return len(self.data)

def get_loader(data, vocab_cap, vocab_tag, batch_size, shuffle, num_workers):
    imgsearch = ImageSearchDataset(vocab_cap, vocab_tag, data)

    data_loader = torch.utils.data.DataLoader(dataset=imgsearch,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=imgsearch.collate_fn)
    return data_loader

def get_loaders(args):
    print("----- Loading Vocab -----")
    vocab_cap = pickle.load(open('vocab_cap.pkl', 'rb'))
    vocab_tag = pickle.load(open('vocab_tag.pkl', 'rb'))
    print(f"vocab cap size: {len(vocab_cap)}, vocab tag cap size: {len(vocab_tag)}")
    print('----- Loading Note -----')

    train = pickle.load(open('train.pkl', 'rb'))
    valid = pickle.load(open('val.pkl', 'rb'))
    test = pickle.load(open('test.pkl', 'rb'))

    #train, valid = train_test_split(train, test_size=0.2, random_state=19)
    print("train size", len(train))
    print("val size", len(valid))
    print("test size", len(test))
    print()
    print('----- Building Loaders -----')
    train_loader = get_loader(train, vocab_cap, vocab_tag, args.batch_size, True, 10)
    valid_loader = get_loader(valid, vocab_cap, vocab_tag, args.batch_size, True, 10)
    test_loader = get_loader(test, vocab_cap, vocab_tag, args.batch_size, False, 10)
    return train_loader, valid_loader, test_loader, vocab_cap, vocab_tag
