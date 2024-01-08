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
from transformer.output_embeddings import get_output_embeddings, OutputEmbedder, PositionalEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv
from tagger import Tagger
OUTPUT_SIZE = 33 * 4 + 5


def calculate_class_frequencies(labels, pad_idx):
    labels_flat = labels.view(-1)
    labels_non_padding = labels_flat[labels_flat != pad_idx]
    class_counts = torch.bincount(labels_non_padding, minlength=OUTPUT_SIZE)
    return class_counts


def eval_epoch(segpos_model, model, loss_function, bert_tokenizer, ctb_ver):
    with open('eval_model.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Acc', 'F1'])

    if ctb_ver == 7:
        eval_set = read_csv('data/CTB7/dev.tsv')
    elif ctb_ver == 9:
        eval_set = read_csv('data/CTB9/dev.tsv')
    else:
        raise FileNotFoundError

    eval_texts, eval_tags, _ = extract_sentences(eval_set)
    eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
    eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    segpos_model.eval()
    tagger = Tagger(segpos_model, 4, 64, 0, 0, 133, 134)
    total_eval_loss = 0
    all_predictions = []
    all_targets = []
    all_masks = []
    batch_count = 1
    with torch.no_grad():
        for batch in eval_dataloader:
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

            # predictions = segpos_model(input_embeddings, batch_attention_masks, batch_output_ids, batch_output_masks)
            # loss = loss_function(predictions.view(-1, OUTPUT_SIZE), batch_output_ids.view(-1))
            predicted_labels = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)
            print(predicted_labels.shape)
            # loss = loss_function(predictions.view(-1, predictions.size(-1)).float(), batch_output_ids.view(-1).long())
            # predicted_labels = torch.argmax(predictions, dim=2)
            batch_output_ids = batch_output_ids
            predicted_labels = predicted_labels.clone() * batch_attention_masks
            batch_output_ids = batch_output_ids.clone() * batch_output_masks
            predicted_labels = predicted_labels.clone().float()
            batch_output_ids = batch_output_ids.clone().float()
            predicted_labels.requires_grad = True
            # total_eval_loss += loss.item()

            predicted_labels_flat = predicted_labels.view(-1)
            batch_output_ids_flat = batch_output_ids.view(-1)
            non_padding_mask = batch_output_ids_flat != 0
            predicted_non_padding = predicted_labels_flat[non_padding_mask]
            targets_non_padding = batch_output_ids_flat[non_padding_mask]
            correct_predictions = (predicted_non_padding == targets_non_padding).sum()
            accuracy = correct_predictions.float() / targets_non_padding.size(0)

            batch_acc = accuracy.item()
            all_predictions.append(predicted_labels_flat[non_padding_mask])

            all_targets.append(batch_output_ids_flat[non_padding_mask])
            all_masks.append(non_padding_mask)

            print(predicted_labels, batch_output_ids)
            # print(f'batch {batch_count} / {math.ceil(len(eval_dataset) / 128)} batch accuracy {batch_acc}')
            print(f'batch {batch_count} / 100 batch accuracy {batch_acc}')

            if batch_count == 100:
                break
            batch_count += 1
    # avg_eval_loss = total_eval_loss / (len(eval_dataset) / 128)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_masks = torch.cat(all_masks)

    f1 = f1_score(all_targets.cpu().numpy(), all_predictions.cpu().numpy(), average='weighted')
    correct_predictions = (all_predictions == all_targets).sum()
    accuracy = correct_predictions.float() / all_targets.size(0)
    accuracy = accuracy.item()

    # print(f"Evaluation Loss: {avg_eval_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    with open('eval_model.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([accuracy, f1])
    return accuracy, f1


def train(ctb_ver: int):
    assert ctb_ver in {7, 9}
    with open(f'model_metrics{ctb_ver}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Batch', 'AvgLoss', 'Loss'])

    print(f'running CTB{ctb_ver}...\n')
    data = read_csv(f'data/CTB{ctb_ver}/train.tsv')

    texts, tags, max_len = extract_sentences(data)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
    output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
    model = load_model('bert-base-chinese')
    encoder = make_encoder()
    decoder = make_decoder()
    output_embedder = OutputEmbedder(768, 768)
    positional_encoder = PositionalEncoder(768, drop_out=0.1, max_len=64)
    segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768,
                                     output_embedder=output_embedder, positional_encoder=positional_encoder)


    dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    num_epochs = 12
    curr_f1 = 0
    combined_parameters = list(model.parameters()) + list(segpos_model.parameters())
    optimizer = AdamW(combined_parameters, lr=5e-5, betas=(0.9, 0.98), eps=10e-9)
    optimizer = ScheduledOptim(optimizer, 1, 768, 250)
    loss_function = CrossEntropyLoss()

    for epoch in range(num_epochs):
        segpos_model.train()
        total_loss = 0
        batch_count = 0
        for batch in dataloader:
            optimizer.zero_grad()

            batch_count += 1
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

            predictions = segpos_model(input_embeddings, batch_attention_masks, batch_output_ids, batch_output_masks)
            predicted_labels = torch.argmax(predictions, dim=2)
            print(predicted_labels)
            # print(batch_output_ids)
            loss = loss_function(predictions.view(-1, predictions.size(-1)).float(),
                                 torch.cat([batch_output_ids[:, 1:], torch.zeros(batch_output_ids.size(0), 1).long()], dim=1).view(-1).long())
            # print(predictions.view(-1, predictions.size(-1)).float())
            print(torch.cat([batch_output_ids[:, 1:], torch.zeros(batch_output_ids.size(0), 1).long()], dim=1).view(-1).long())

            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            with open(f'model_metrics{ctb_ver}', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_count, total_loss/batch_count, loss.item()])

            print(f'epoch: {epoch}, batch: {batch_count} / {math.ceil(len(dataset) / 128)}, total_loss: {total_loss}, avg_loss: {total_loss/batch_count}', f'batch_loss: {loss.item()}')

        acc, f1 = eval_epoch(segpos_model, model, loss_function, bert_tokenizer, ctb_ver)
        if f1 > curr_f1:
            torch.save(segpos_model.state_dict(), f'./model_state_dict{ctb_ver}')
            curr_f1 = f1
    return segpos_model, model


segpos_model, bert_model = train(9)
