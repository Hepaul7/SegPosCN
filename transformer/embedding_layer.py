"""
Embeddings layer
"""
import torch
from transformers import BertTokenizer, BertModel
from typing import List
from constants import *

import csv

BERT_MODEL_NAME = 'bert-base-chinese'


def read_csv(path: str) -> List[List[str]]:
    """ Converts CSV data into a list of list
    :param path: path to the CSV file containing data
    to be read
    :return: List of Lists, containing character and
    labels, for example:
    [
    ['洲', 'S-NN'],
    ['冠', 'B-NN'],
    // ... more lines ...
    ]
    Note: May contain empty lists
    """
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [line for line in reader]
    return data


# def extract_sentences(data: List[List[str]]) -> (List[str], int):
#     """ Breaks down some text into individual sentence
#     :param data: List of lists containing character and a label
#     :return: List of strings, where each string is a Chinese sentence
#     """
#     sentences = []
#     curr = ''
#     max_len = 0
#     for ls in data:
#         if len(ls) > 0:
#             curr += ls[0]
#             continue
#         if curr != '':
#             sentences.append(curr)
#             if len(curr) > max_len:
#                 max_len = len(curr)
#             curr = ''
#
#     return sentences, max_len

def extract_sentences(data: List[List[str]]) -> (List[List[str]], int):
    """ Breaks down some text into individual sentence
    :param data: List of lists containing character and a label
    :return: List of strings, where each string is a Chinese sentence
    """
    sentences = []
    curr = []
    max_len = 0
    for ls in data:
        if len(ls) > 0:
            curr.append(ls[0])
            continue
        if curr:
            sentences.append(curr)
            if len(curr) > max_len:
                max_len = len(curr)
            curr = []

    return sentences, max_len


def extract_labels(data: List[List[str]]) -> (List[List[str]], int):
    """ Breaks down some text into their tags
    :param data:
    :return: List of tags
    """
    tags = []
    curr = []
    max_len = 0
    for ls in data:
        if len(ls) > 0:
            curr.append(ls[1])
            continue
        if curr:
            tags.append(curr)
            if len(curr) > max_len:
                max_len = len(curr)
            curr = []
    return tags, max_len


def prepare_data(data: List[List[str]], max_len: int) -> List[List[str]]:
    """
    Add padding, BOS, EOS
    :param data:
    :param max_len:
    :return:
    """
    for sentence in data:
        sentence.insert(0, BOS_WORD)
        sentence.append(EOS_WORD)
        while len(sentence) < max_len:
            sentence.append(PAD_WORD)
    return data


def prepare(data: List[List[str]], tokenizer: BertTokenizer, max_length: int) -> [torch.Tensor, torch.Tensor]:
    """ Prepare texts into a form that can be fed into a
    pretrained transformer (Chinese-BERT)
    :param data: List of strings, where each string is a Chinese sentence
    :param tokenizer: Tokenizer
    :param max_length: max length of sentence within batch
    :return: input_ids: tokenized representation of text,
             attention_masks: tensors of 1s and 0s
    """
    joined_sentences = [''.join(sentence) for sentence in data]

    encoding = tokenizer(
        joined_sentences,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Extract input_ids and attention_masks
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']

    return input_ids, attention_masks


def load_model(model_name):
    model = BertModel.from_pretrained(model_name)
    return model


def get_bert_embeddings(model, input_ids, attention_masks):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states


# Note to self, perhaps adapt MCCWS 3 embeddings?
# class PositionalEncoder(nn.Module):
#     """
#     I DONT NEED THIS BERT EMBEDDINGS ALREADY HAVE POSITIONAL INFORMATION AGHGHrhgRHJGJHFG
#
#     Since transformer models contain no recurrence or convolution, we need
#     to inject some information about relative or absolute position of the
#     sequence (Vaswani et al. 2017)
#
#     Add positional encodings to the input embeddings before the encoder
#     layer.
#
#     Implementation of this class is based on section 3.5 of Attention is
#     all you need (Vaswani et al. 2017)
#     """
#     def __init__(self, d_model: int, drop_out: float, max_len: int = 5000):
#         """
#         :param d_model: dimension of the model, if BERT it would be 768
#         :param drop_out: drop out rate
#         :param max_len: max length
#         """
#         super(PositionalEncoder, self).__init__()
#         self.dropout = nn.Dropout(p=drop_out)
#
#         pe = torch.zeros(max_len, d_model).float()
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         # Apply sine function to even indices (pos, 2i)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         # Apply cosine function to odd indices (pos, 2i + 1)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         # For batch processing
#         pe = pe.unsqueeze(0)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.tensor):
#         x += self.pe[:, x.size(0), :]
#         return self.dropout(x)
#
#

# def main():
#     data = read_csv('data/CTB7/dev.tsv')
#     texts = extract_sentences(data)
#     bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     input_ids, attention_masks = prepare(texts, bert_tokenizer)
#
#     model = load_model('bert-base-chinese')
#     embeddings = get_bert_embeddings(model, input_ids, attention_masks)
#     p_e = PositionalEncoder(d_model=768, drop_out=0.1)
#     embeddings = p_e(embeddings)
#     return embeddings

