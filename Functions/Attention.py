import torch
import math

def attention_matrix(q, k, v, dropout=None, mask=None):

    d_k = q.size(-1) 
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # Durch eine Maske geblockte Einträge erhalten ein sehr hohen negativen Attention Score
    att_matrix = scores.softmax(dim=-1)
    #print(att_matrix.shape)
    if dropout is not None:
        att_matrix = dropout(att_matrix) # Dropout beim Training
    return att_matrix

def attention(q, k, v, dropout=None, mask=None):
    att_matrix = attention_matrix(q, k, v, dropout=dropout, mask=mask)
    #print( torch.matmul(att_matrix, v).shape)
    return torch.matmul(att_matrix, v) 