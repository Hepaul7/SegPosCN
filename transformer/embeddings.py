"""
Embeddings layer
"""
import torch
from transformers import BertTokenizer, BertModel
from typing import List
from constants import *

import csv

BERT_MODEL_NAME = 'bert-base-chinese'

POS_tags = [
    "VA", "VC", "VE", "VV", "NR", "NT", "NN", "LC", "PN", "DT", "CD", "OD",
    "M", "AD", "P", "CC", "CS", "DEC", "DEG", "DER", "DEV", "SP", "AS", "ETC",
    "MSP", "IJ", "ON", "PU", "JJ", "FW", "LB", "SB", "BA"
]

BMES_tags = ["B", "M", "E", "S"]
PAD = 0
UNK = 100
CLS = 133
SEP = 134
MASK = 103
LIMIT = 62

tag_to_id = {}
id_to_tag = {}

for i, pos_tag in enumerate(POS_tags):
    id_to_tag['PAD'] = PAD
    id_to_tag['CLS'] = CLS
    id_to_tag['SEP'] = SEP
    tag_to_id[PAD] = 'PAD'
    tag_to_id[CLS] = 'CLS'
    tag_to_id[SEP] = 'SEP'
    for j, bmes_tag in enumerate(BMES_tags):
        tag = f"{bmes_tag}-{pos_tag}"
        tag_id = i * len(BMES_tags) + j + 1
        tag_to_id[tag] = tag_id
        id_to_tag[tag_id] = tag
print(len(tag_to_id.keys()))
print(tag_to_id)


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


def extract_sentences(data: List[List[str]]) -> (List[str], List[List[str]], int):
    """ Breaks down some text into individual sentence
    :param data: List of lists containing character and a label
    :return: List of strings, where each string is a Chinese sentence
    """
    sentences = []
    tags = []
    max_len = 0
    curr = ''
    curr_out = []
    total_len = 0
    count_lim = 0
    for ls in data:
        if len(ls) > 0:
            if any(substring in ls[1] for substring in {'URL', 'X', 'BULLET'}):
                continue
            curr += ls[0]
            if 'SHORT' in ls[1]:
                parts = tag.split('-')
                ls[1] = '-'.join(parts[:2])
            curr_out.append(ls[1])
            continue
        if curr != '':
            if len(curr) > max_len:
                max_len = len(curr)
                # print(max_len)
            if len(curr) < LIMIT:
                sentences.append(curr)
                tags.append(curr_out)
                total_len += len(curr)
                count_lim += 1
            curr = ''
            curr_out = []
    print(count_lim)
    print(f'avg_len: {total_len // len(sentences)}') # is around 37
    return sentences, tags, max_len


def extract_characters_from_text(text: str) -> List[str]:
    """ Break down sentences into individual characters from text
    :param text: string of texts
    :return: List of characters
    """
    return [char for char in text]


def prepare(texts: List[str], tokenizer: BertTokenizer, max_len: int) -> [torch.Tensor, torch.Tensor]:
    """ Prepare texts into a form that can be fed into a
    pretrained transformer (Chinese-BERT)
    :param max_len:
    :param texts: List of strings, where each string is a Chinese sentence
    :param tokenizer: Tokenizer
    :return: input_ids: tokenized representation of text,
             attention_masks: tensors of 1s and 0s
    """
    input_ids = []
    attention_masks = []
    vocab = tokenizer.get_vocab()

    oov, total_len = 0, 0
    for text in texts:
        for char in text:
            if char not in vocab.keys():
                oov += 1
            total_len += 1
        char_tokens = extract_characters_from_text(text)
        encoded_dict = tokenizer.encode_plus(
            char_tokens,
            add_special_tokens=True,
            max_length=LIMIT + 2,
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
    print(f'oov rate: {oov / total_len} oov: {oov}, total: {total_len}')
    print(vocab)
    print(input_ids[0], attention_masks[0])
    return input_ids, attention_masks


def prepare_outputs(tags, max_len):
    """
    Prepare outputs by encoding each tag into an index in 4 x 33
    :param tags:
    :param max_len:
    :return:
    """
    # TODO, FIX FROM 0x33 to 4x33 + 3
    outputs = torch.zeros(len(tags), LIMIT + 2, dtype=torch.long)
    mask = torch.zeros(len(tags), LIMIT + 2, dtype=torch.long)
    for x in range(len(tags)):
        if len(tags[x]) >= LIMIT:
            continue
        outputs[x][0], mask[x][0] = CLS, 1
        outputs[x][len(tags[x]) + 1], mask[x][len(tags[x]) + 1] = SEP, 1
        for y in range(max_len):
            if y < len(tags[x]):
                outputs[x][y + 1], mask[x][y + 1] = tag_to_id[tags[x][y]], 1
    print(outputs[0], mask[0])
    return outputs, mask


def load_model(model_name):
    model = BertModel.from_pretrained(model_name)
    return model


def get_bert_embeddings(model, input_ids, attention_masks):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

