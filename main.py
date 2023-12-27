# from training import *
#
# #####
# # This file is for testing (for now)
# #####
# print('running CTB7...\n')
# data = read_csv('data/CTB7/train.tsv')
#
# texts, tags, max_len = extract_sentences(data)
#
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#
# input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
# output_ids, output_masks = prepare_outputs(tags, 64)  # includes BOS/EOS
# bert_model = load_model('bert-base-chinese')
#
# # input_embeddings = get_bert_embeddings(model, input_ids, attention_masks)
# # output_embeddings = get_bert_embeddings(model, decoder_input_ids, decoder_attention_mask)
# encoder = make_encoder()
# decoder = make_decoder()
# segpos_model = CWSPOSTransformer(encoder, decoder, output_size=33 * 4 + 2, d_model=768)
#
# # change to embeddings later
# dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# num_epochs = 100
# optimizer = AdamW(segpos_model.parameters(), lr=5e-5)
# loss_function = CrossEntropyLoss()
#
# # load eval set
# eval_set = read_csv('data/CTB7/dev.tsv')
#
# eval_texts, eval_tags, _ = extract_sentences(eval_set)
# eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
# eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
# eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
# eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
#
#
# for epoch in range(num_epochs):
#     avg_train_loss = train_epoch(segpos_model, bert_model, dataloader, loss_function, optimizer)
#     print(f"Epoch {epoch+1} Training Loss: {avg_train_loss}")
#
#     # Evaluate after each epoch
#     avg_eval_loss, accuracy, f1 = evaluate_model(segpos_model, bert_model, eval_dataloader, loss_function)
#     print(f"Epoch {epoch+1} Evaluation Metrics:\n Loss: {avg_eval_loss}, Accuracy: {accuracy}, F1 Score: {f1}")
#
#
import math

from transformers import AdamW
from optimizer import *

from transformer.embeddings import *
from transformer.encoder import make_encoder
from transformer.decoder import make_decoder
from models import CWSPOSTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader, TensorDataset
from transformer.output_embeddings import get_output_embeddings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

OUTPUT_SIZE = 33 * 4 + 3

def eval_epoch(segpos_model, model, loss_function, bert_tokenizer):
    eval_set = read_csv('data/CTB7/dev.tsv')

    eval_texts, eval_tags, _ = extract_sentences(eval_set)
    eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
    eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
    eval_dataloader = DataLoader(eval_dataset, batch_size=33, shuffle=False)

    segpos_model.eval()  # Set the model to evaluation mode
    total_eval_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # No gradients needed for evaluation
        for batch in eval_dataloader:
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)
            output_embeddings = get_output_embeddings(batch_output_ids)
            print(input_embeddings[0])
            print(output_embeddings[0])

            predictions = segpos_model(input_embeddings, batch_attention_masks, output_embeddings, batch_output_masks)
            loss = loss_function(predictions.view(-1, OUTPUT_SIZE), batch_output_ids.view(-1))
            total_eval_loss += loss.item()

            # Flatten the predictions and true labels
            predictions = predictions.view(-1, OUTPUT_SIZE)
            batch_output_ids = batch_output_ids.view(-1)

            # Convert predictions to actual label indices
            predicted_labels = torch.argmax(predictions, dim=1)

            # Append predictions and true labels for metric calculation
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(batch_output_ids.cpu().numpy())

    # Calculate average loss over all batches
    avg_eval_loss = total_eval_loss / len(eval_dataloader)

    # Remove padding token label IDs for metric calculation
    mask = [label_id < OUTPUT_SIZE for label_id in all_true_labels]  # Adjust the number based on your label range
    all_predictions = [pred for pred, m in zip(all_predictions, mask) if m]
    all_true_labels = [true_label for true_label, m in zip(all_true_labels, mask) if m]

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')  # 'weighted' accounts for label imbalance

    print(f"Evaluation Loss: {avg_eval_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")


def train():
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
    segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768)

    # change to embeddings later
    dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
    dataloader = DataLoader(dataset, batch_size=35, shuffle=False)

    num_epochs = 10  # start: 19:50
    optimizer = AdamW(segpos_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=10e-9)
    optimizer = ScheduledOptim(optimizer, 1, 768, 4000)
    loss_function = CrossEntropyLoss()

    for epoch in range(num_epochs):
        segpos_model.train()
        total_loss = 0
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)
            # print(input_embeddings.shape, batch_attention_masks.shape)
            output_embeddings = get_output_embeddings(batch_output_ids)
            # print(output_embeddings.shape, batch_output_masks.shape)
            # print(input_embeddings.shape)
            predictions = segpos_model(input_embeddings, batch_attention_masks, output_embeddings, batch_output_masks)
            # TODO: predictions need to be decoded
            # print(predictions.shape)
            # print(predictions.view(-1, OUTPUT_SIZE).shape, batch_output_ids.shape)
            # TODO: need to convert batch_output_ids into tags
            batch_prediction_ids = torch.zeros(batch_output_ids.shape[0], batch_output_ids.shape[1])
            for z in range(predictions.shape[2]):
                batch_prediction_ids = torch.max(predictions, 2).indices
                batch_prediction_ids[batch_prediction_ids != 0] += 103
            predictions = F.softmax(predictions)
            batch_output_probabilities = F.one_hot(batch_output_ids, num_classes=4 * 33 + 3).float()
            loss = loss_function(predictions, batch_output_probabilities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            print(f'epoch: {epoch}, batch: {batch_count} / {math.ceil(len(dataset) / 35)}, total_loss: {total_loss}, avg_loss: {total_loss/batch_count}', f'batch_los: {loss.item()}')

        eval_epoch(segpos_model, model, loss_function, bert_tokenizer)
    return segpos_model, model


segpos_model, bert_model = train()
