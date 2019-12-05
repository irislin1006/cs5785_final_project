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
    def __init__(self, vocab_size, emb_dim, img_fea_size, dropout):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(Ingres2Recipe, self).__init__()
        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.linear_img1 = nn.Linear(img_fea_size, 200)
        self.linear_img2 = nn.Linear(200, 100)
        self.linear_i1 = nn.Linear(emb_dim, 200)
        self.linear_i2 = nn.Linear(200, 100)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(200)

    def forward(self, img_fea, captions, c_lengths):
        """
        @param recipes, ingres: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param r_length, i_length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        imgf = F.relu(self.linear_img1(img_fea))
        imgf = self.batchnorm(imgf)
        imgf = self.dropout(imgf)
        imgf = self.linear_img2(imgf)

        cap = self.embed(captions)
        cap = self.dropout(cap)
        cap = torch.sum(cap, dim=1)
        cap /= c_lengths.float() + 1e-10
        cap = F.relu(self.linear_i1(cap))
        cap = self.dropout(cap)
        cap = self.linear_i2(cap)

        return imgf, cap
