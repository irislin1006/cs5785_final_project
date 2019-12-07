"""
this code is modified from https://github.com/ksenialearn/bag_of_words_pytorch/blob/master/bag_of_words-master-FINAL.ipynb
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Ingres2Recipe(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_cap_size, vocab_tag_size, emb_dim, img_fea_size, dropout):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(Ingres2Recipe, self).__init__()
        # pay attention to padding_idx

        self.linear_i1 = nn.Linear(img_fea_size, 500)
        self.linear_i2 = nn.Linear(500, 200)
        self.linear_i3 = nn.Linear(200, vocab_tag_size)

        self.dropout = nn.Dropout(dropout)
        self.bn_img = nn.BatchNorm1d(img_fea_size)
        self.bn_1 = nn.BatchNorm1d(500)
        self.bn_2 = nn.BatchNorm1d(200)

    def forward(self, img_fea):
        """
        @param recipes, ingres: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param r_length, i_length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        img = self.bn_img(img_fea)
        img = self.bn_1(F.elu(self.linear_i1(img)))
        img = self.dropout(img)
        img = self.bn_2(F.elu(self.linear_i2(img)))
        img = self.dropout(img)
        logit = self.linear_i3(img)

        return logit
