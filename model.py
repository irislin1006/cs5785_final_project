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
        self.embed = nn.Embedding(vocab_cap_size, emb_dim, padding_idx=0)
        self.embed_tag = nn.Embedding(vocab_tag_size, emb_dim, padding_idx=0)
        hidden = 400
        self.linear_img1 = nn.Linear(img_fea_size, 500)
        self.linear_img2 = nn.Linear(500, hidden)
        self.linear_i1 = nn.Linear(emb_dim, 500)
        self.linear_i2 = nn.Linear(500, hidden)
        self.linear_t1 = nn.Linear(emb_dim, 500)
        self.linear_t2 = nn.Linear(500, hidden)
        self.linear_ti = nn.Linear(2*hidden, hidden)
        self.linear_ti2 = nn.Linear(hidden, hidden)


        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(img_fea_size)
        self.bn_img = nn.BatchNorm1d(hidden)
        self.bn_cap = nn.BatchNorm1d(hidden)
        self.bn_tag = nn.BatchNorm1d(hidden)
        self.bn_ti = nn.BatchNorm1d(hidden)

    def forward(self, img_fea, captions, tags, c_lengths, t_lengths):
        """
        @param recipes, ingres: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param r_length, i_length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        img_fea = self.batchnorm(img_fea)
        imgf = F.elu(self.linear_img1(img_fea))
        #imgf = self.batchnorm(imgf)
        imgf = self.dropout(imgf)
        imgf = self.bn_img(self.linear_img2(imgf))

        cap = self.embed(captions)
        cap = self.dropout(cap)
        cap = torch.sum(cap, dim=1)
        cap /= c_lengths.float() + 1e-10
        cap = F.elu(self.linear_i1(cap))
        cap = self.dropout(cap)
        cap = self.bn_cap(self.linear_i2(cap))

        tags = self.embed_tag(tags)
        tags = self.dropout(tags)
        tags = torch.sum(tags, dim=1)
        tags /= t_lengths.float() + 1e-10
        tags = F.relu(self.linear_t1(tags))
        tags = self.dropout(tags)
        tags = self.bn_cap(self.linear_t2(tags))

        tag_img = torch.cat([imgf, tags], dim=1)
        tag_img = F.relu(self.linear_ti(tag_img))
        tag_img = self.dropout(tag_img)
        tag_img = self.bn_ti(self.linear_ti2(tag_img))
        return imgf, cap, tags, imgf
