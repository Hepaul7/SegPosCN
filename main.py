from transformers import AdamW

from transformer.input_embeddings import *
from transformer.encoder import make_encoder
from transformer.decoder import make_decoder
from models import CWSPOSTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader, TensorDataset

#####
# This file is for testing (for now)
#####
print('running CTB7...\n')
data = read_csv('data/CTB7/train.tsv')

texts, tags, max_len = extract_sentences(data)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
output_ids, output_masks = prepare_outputs(tags, 64)  # includes BOS/EOS
model = load_model('bert-base-chinese')

# input_embeddings = get_bert_embeddings(model, input_ids, attention_masks)
# output_embeddings = get_bert_embeddings(model, decoder_input_ids, decoder_attention_mask)
encoder = make_encoder()
decoder = make_decoder()
segpos_model = CWSPOSTransformer(encoder, decoder, output_size=33 * 4 + 2, d_model=768)

# change to embeddings later
dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

num_epochs = 5
optimizer = AdamW(segpos_model.parameters(), lr=5e-5)
loss_function = CrossEntropyLoss()

for epoch in range(num_epochs):
    segpos_model.train()
    total_loss = 0
    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch

        input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

        print(input_embeddings.shape)
        predictions = segpos_model(input_embeddings, batch_attention_masks, batch_output_ids, batch_output_masks)

        loss = loss_function(predictions.view(-1, 33 * 4 + 2), batch_output_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

