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

        self.linear_i1 = nn.Linear(emb_dim, 500)
        self.linear_i2 = nn.Linear(500, 1000)
        self.linear_i3 = nn.Linear(1000, img_fea_size)

        self.dropout = nn.Dropout(dropout)
        self.bn_1 = nn.BatchNorm1d(500)
        self.bn_2 = nn.BatchNorm1d(1000)

    def forward(self, img_fea, captions, tags, c_lengths, t_lengths):
        """
        @param recipes, ingres: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param r_length, i_length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """

        cap = self.embed(tags)
        cap = self.dropout(cap)
        cap = torch.sum(cap, dim=1)
        cap /= c_lengths.float() + 1e-10
        cap = self.bn_1(F.elu(self.linear_i1(cap)))
        cap = self.dropout(cap)
        cap = self.bn_2(F.elu(self.linear_i2(cap)))
        cap = self.dropout(cap)
        cap = self.linear_i3(cap)

        tags = self.embed_tag(tags)
        tags = self.dropout(tags)
        tags = torch.sum(tags, dim=1)
        #tags /= t_lengths.float() + 1e-10
        #tags = F.relu(self.linear_i1(tags))
        #tags = self.dropout(tags)

        #tag_img = torch.cat([imgf, tags], dim=1)
        #tag_img = F.relu(self.linear_ti(tag_img))
        #tag_img = self.dropout(tag_img)
        #tag_img = self.bn_ti(self.linear_ti2(tag_img))
        return cap, cap, tags, cap
