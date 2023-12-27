from transformers import AdamW

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


def train_epoch(model, bert_model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Unpack the batch data
        batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch

        # Get embeddings
        input_embeddings = get_bert_embeddings(bert_model, batch_input_ids, batch_attention_masks)
        output_embeddings = get_output_embeddings(batch_output_ids)

        # Forward pass
        predictions = model(input_embeddings, batch_attention_masks, output_embeddings, batch_output_masks)

        # Compute loss
        loss = loss_function(predictions.view(-1, 33 * 4 + 2), batch_output_ids.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, bert_model, dataloader, loss_function):
    model.eval()
    total_eval_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
            input_embeddings = get_bert_embeddings(bert_model, batch_input_ids, batch_attention_masks)
            output_embeddings = get_output_embeddings(batch_output_ids)

            predictions = model(input_embeddings, batch_attention_masks, output_embeddings, batch_output_masks)
            loss = loss_function(predictions.view(-1, 33 * 4 + 2), batch_output_ids.view(-1))
            total_eval_loss += loss.item()

            # Process predictions for metrics
            predicted_labels = torch.argmax(predictions, dim=1).view(-1)
            batch_output_ids = batch_output_ids.view(-1)

            # Append predictions and true labels for metric calculation
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(batch_output_ids.cpu().numpy())

    # Filter out padding token label IDs
    mask = [label_id < 33 * 4 + 2 for label_id in all_true_labels]
    all_predictions = [pred for pred, m in zip(all_predictions, mask) if m]
    all_true_labels = [true_label for true_label, m in zip(all_true_labels, mask) if m]

    # Calculate metrics
    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')

    return avg_eval_loss, accuracy, f1



