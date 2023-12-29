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
OUTPUT_SIZE = 33 * 4 + 5


def calculate_class_frequencies(labels, pad_idx):
    # Flatten the labels tensor to a 1D array
    labels_flat = labels.view(-1)

    # Filter out padding indices
    labels_non_padding = labels_flat[labels_flat != pad_idx]

    # Count occurrences of each class
    class_counts = torch.bincount(labels_non_padding, minlength=OUTPUT_SIZE)
    return class_counts


def eval_epoch(segpos_model, model, loss_function, bert_tokenizer):
    with open('eval_model.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['AvgLoss', 'Acc', 'F1'])

    eval_set = read_csv('data/CTB7/dev.tsv')

    eval_texts, eval_tags, _ = extract_sentences(eval_set)
    eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
    eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=True)

    segpos_model.eval()  # Set the model to evaluation mode
    total_eval_loss = 0
    all_predictions = []
    all_targets = []
    all_masks = []
    batch_count = 1
    with torch.no_grad():  # No gradients needed for evaluation
        for batch in eval_dataloader:
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

            predictions = segpos_model(input_embeddings, batch_attention_masks, batch_output_ids, batch_output_masks)
            # loss = loss_function(predictions.view(-1, OUTPUT_SIZE), batch_output_ids.view(-1))

            loss = loss_function(predictions.view(-1, predictions.size(-1)).float(), batch_output_ids.view(-1).long())
            predicted_labels = torch.argmax(predictions, dim=2)
            batch_output_ids = batch_output_ids
            predicted_labels = predicted_labels.clone() * batch_attention_masks
            batch_output_ids = batch_output_ids.clone() * batch_output_masks
            predicted_labels = predicted_labels.clone().float()
            batch_output_ids = batch_output_ids.clone().float()
            predicted_labels.requires_grad = True
            total_eval_loss += loss.item()

            predicted_labels_flat = predicted_labels.view(-1)
            batch_output_ids_flat = batch_output_ids.view(-1)

            # Create a mask for non-zero (non-padding) elements
            non_padding_mask = batch_output_ids_flat != 0

            # Apply the mask to filter out padding tokens
            predicted_non_padding = predicted_labels_flat[non_padding_mask]
            targets_non_padding = batch_output_ids_flat[non_padding_mask]

            # Calculate the number of correct predictions (only for non-padding tokens)
            correct_predictions = (predicted_non_padding == targets_non_padding).sum()

            # Calculate accuracy
            accuracy = correct_predictions.float() / targets_non_padding.size(0)
            # Convert to a Python number for reporting
            batch_acc = accuracy.item()
            all_predictions.append(predicted_labels_flat[non_padding_mask])

            all_targets.append(batch_output_ids_flat[non_padding_mask])
            all_masks.append(non_padding_mask)

            print(predicted_labels, batch_output_ids)
            print(f'batch {batch_count} / {math.ceil(len(eval_dataset) / 128)} batch accuracy {batch_acc}')
            batch_count += 1
        # Calculate average loss over all batches
    avg_eval_loss = total_eval_loss / (len(eval_dataset) / 128)

    # Remove padding token label IDs for metric calculation
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_masks = torch.cat(all_masks)

    f1 = f1_score(all_targets.cpu().numpy(), all_predictions.cpu().numpy(), average='weighted')
    correct_predictions = (all_predictions == all_targets).sum()
    accuracy = correct_predictions.float() / all_targets.size(0)
    accuracy = accuracy.item()

    print(f"Evaluation Loss: {avg_eval_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    with open('eval_model.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([avg_eval_loss, accuracy, f1])


def train():
    with open('model_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Batch', 'AvgLoss', 'Loss'])

    print('running CTB7...\n')
    data = read_csv('data/CTB7/train.tsv')

    texts, tags, max_len = extract_sentences(data)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
    output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
    model = load_model('bert-base-chinese')
    for param in model.parameters():
        param.requires_grad = False

    class_counts = calculate_class_frequencies(output_ids, PAD)
    class_weights = 1.0 / class_counts
    class_weights[PAD] = 0
    encoder = make_encoder()
    decoder = make_decoder()
    output_embedder = OutputEmbedder(768, 768)
    positional_encoder = PositionalEncoder(768, drop_out=0.1, max_len=64)
    segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768,
                                     output_embedder=output_embedder, positional_encoder=positional_encoder)

    dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    num_epochs = 10
    optimizer = AdamW(segpos_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=10e-9)
    optimizer = ScheduledOptim(optimizer, 1, 768, 250)
    loss_function = CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):
        segpos_model.train()
        total_loss = 0
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)
            predictions = segpos_model(input_embeddings, batch_attention_masks, batch_output_ids, batch_output_masks)
            predicted_labels = torch.argmax(predictions, dim=2)
            print(predicted_labels)
            print(batch_output_ids)
            loss = loss_function(predictions.view(-1, predictions.size(-1)).float(), batch_output_ids.view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            with open('model_metrics.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_count, total_loss/batch_count, loss.item()])
            print(f'epoch: {epoch}, batch: {batch_count} / {math.ceil(len(dataset) / 128)}, total_loss: {total_loss}, avg_loss: {total_loss/batch_count}', f'batch_loss: {loss.item()}')
        eval_epoch(segpos_model, model, loss_function, bert_tokenizer)
        torch.save(segpos_model.state_dict(), './model_state_dict.pth')
    return segpos_model, model


segpos_model, bert_model = train()