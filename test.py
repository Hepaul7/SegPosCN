import torch
from torch.utils.data import TensorDataset, DataLoader

from transformer.embeddings import *
from transformer.encoder import *
from transformer.decoder import *
from transformer.output_embeddings import *
from models import *
import random
OUTPUT_SIZE = 33 * 4 + 5

data = read_csv('data/CTB7/train.tsv')

texts, tags, max_len = extract_sentences(data)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
model = load_model('bert-base-chinese')
for param in model.parameters():
    param.requires_grad = False

encoder = make_encoder()
decoder = make_decoder()
output_embedder = OutputEmbedder(768, 768)
positional_encoder = PositionalEncoder(768, drop_out=0.1, max_len=64)
segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768,
                                 output_embedder=output_embedder, positional_encoder=positional_encoder)

print('loading state')
segpos_model.load_state_dict(torch.load('model_state_dict.pth'))
segpos_model.eval()
eval_set = read_csv('data/CTB7/dev.tsv')
#
eval_texts, eval_tags, _ = extract_sentences(eval_set)
# print(eval_texts)
eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=True)

#
total_eval_loss = 0
all_predictions = []
all_targets = []
all_masks = []
batch_count = 1
with torch.no_grad():  # No gradients needed for evaluation
    for batch in eval_dataloader:
        batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
        new = batch_output_ids.clone()
        # new = torch.zeros_like(batch_output_ids)
        # new[:, :] = 0
        # new[:, 1] = 135
        # new[:, 0] = 133

        input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)
        predictions = segpos_model(input_embeddings, batch_attention_masks, new, batch_output_masks)
        # loss = loss_function(predictions.view(-1, OUTPUT_SIZE), batch_output_ids.view(-1))
        batch_output_ids = batch_output_ids
        predicted_labels = torch.argmax(predictions, dim=2)

        predicted_labels = predicted_labels.clone() * batch_attention_masks
        batch_output_ids = batch_output_ids.clone() * batch_output_masks
        predicted_labels = predicted_labels.clone().float()
        batch_output_ids = batch_output_ids.clone().float()
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
        print(predicted_labels, batch_output_ids)
        print(accuracy)

# with torch.no_grad():
#     # my_text = ['晚上看星星白天睡大觉，结果第二天忘记去学校了。']
#     my_text = [eval_texts[0]]
#     input_ids, mask = prepare(my_text, bert_tokenizer, 64)
#     dummy_outputs = torch.zeros_like(input_ids)
#     # for x in range(1, len(my_text[0])):
#     #     dummy_outputs[x] = random.randint(1, 134)
#     dummy_outputs[:, 0] = 133
#     dummy_outputs[:, 1] = 135
#     # dummy_outputs[len(my_text[0])] = 8
#     # print(dummy_outputs.unsqueeze(0))
#     # dummy_outputs[:, len(my_text[0])] = 134
#     input_embeddings = get_bert_embeddings(model, input_ids, mask)
#
#     predictions = segpos_model(input_embeddings, mask, dummy_outputs.unsqueeze(0), mask)
#     predicted_labels = torch.argmax(predictions, dim=2)
#     print(predicted_labels)
