"""
Embeddings layer
"""
import torch
from transformers import BertTokenizer, BertModel
from typing import List, Union

import csv
import torch.nn as nn
import math

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


def extract_sentences(data: List[List[str]]) -> List[str]:
    """ Breaks down some text into individual sentence
    :param data: List of lists containing character and a label
    :return: List of strings, where each string is a Chinese sentence
    """
    sentences = []
    curr = ''
    for ls in data:
        if len(ls) > 0:
            curr += ls[0]
            continue
        if curr != '':
            sentences.append(curr)
            curr = ''

    return sentences


def extract_characters_from_text(text: str) -> List[str]:
    """ Break down sentences into individual characters from text
    :param text: string of texts
    :return: List of characters
    """
    return [char for char in text]


def prepare(texts: List[str], tokenizer: BertTokenizer) -> [torch.Tensor, torch.Tensor]:
    """ Prepare texts into a form that can be fed into a
    pretrained transformer (Chinese-BERT)
    :param texts: List of strings, where each string is a Chinese sentence
    :param tokenizer: Tokenizer
    :return: input_ids: tokenized representation of text,
             attention_masks: tensors of 1s and 0s
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        char_tokens = extract_characters_from_text(text)
        encoded_dict = tokenizer.encode_plus(
            char_tokens,
            add_special_tokens=True,
            max_length=64,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            is_split_into_words=False
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

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
class PositionalEncoder(nn.Module):
    """
    Since transformer models contain no recurrence or convolution, we need
    to inject some information about relative or absolute position of the
    sequence (Vaswani et al. 2017)

    Add positional encodings to the input embeddings before the encoder
    layer.

    Implementation of this class is based on section 3.5 of Attention is
    all you need (Vaswani et al. 2017)
    """
    def __init__(self, d_model: int, drop_out: float, max_len: int = 5000):
        """
        :param d_model: dimension of the model, if BERT it would be 768
        :param drop_out: drop out rate
        :param max_len: max length
        """
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine function to even indices (pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine function to odd indices (pos, 2i + 1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # For batch processing
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor):
        x += self.pe[:, x.size(0), :]
        return self.dropout(x)


def main():
    data = read_csv('data/CTB7/dev.tsv')
    texts = extract_sentences(data)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    input_ids, attention_masks = prepare(texts, bert_tokenizer)

    model = load_model('bert-base-chinese')
    embeddings = get_bert_embeddings(model, input_ids, attention_masks)
    return embeddings

