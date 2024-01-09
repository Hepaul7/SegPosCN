from torch.utils.data import TensorDataset, DataLoader
from transformer.encoder import *
from transformer.decoder import *
from models import *
from tagger import Tagger
from sklearn.metrics import f1_score


OUTPUT_SIZE = 33 * 4 + 5


data = read_csv('data/CTB7/dev.tsv')

texts, tags, max_len = extract_sentences(data)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
model = load_model('bert-base-chinese')
# model.load_state_dict(torch.load('bert_state_dict9-2'))
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
tagger = Tagger(segpos_model, 4, 64, 0, 0, 133, 134)

eval_set = read_csv('data/CTB7/dev.tsv')
#
eval_texts, eval_tags, _ = extract_sentences(eval_set)
# print(eval_texts)
eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
all_predictions = []
all_targets = []
all_masks = []
batch_count = 1

with torch.no_grad():  # No gradients needed for evaluation
    total_batch = 0
    total_cws = 0
    total_pos = 0
    all_cws_pred = []
    all_pos_pred = []
    all_cws_targ = []
    all_pos_targ = []
    for batch in eval_dataloader:
        batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
        batch_output_ids_flat = batch_output_ids.view(-1)
        non_padding_mask = batch_output_ids_flat != 0
        if len(batch_output_ids_flat[non_padding_mask]) < 40:
            continue
        print(len(batch_output_ids_flat[non_padding_mask]))
        input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

        # tagger._get_init_state(input_embeddings, batch_attention_masks, batch_output_masks)
        # predictions = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)
        predicted_labels = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)

        # loss = loss_function(predictions.view(-1, predictions.size(-1)).float(), batch_output_ids.view(-1).long())
        # predicted_labels = torch.argmax(predictions, dim=2)
        batch_output_ids = batch_output_ids
        # predicted_labels = predicted_labels.clone() * batch_attention_masks
        # batch_output_ids = batch_output_ids.clone() * batch_output_masks
        predicted_labels = predicted_labels.clone().float()
        batch_output_ids = batch_output_ids.clone().float()
        predicted_labels.requires_grad = False
        # total_eval_loss += loss.item()

        predicted_labels_flat = predicted_labels.view(-1)
        batch_output_ids_flat = batch_output_ids.view(-1)
        non_padding_mask = batch_output_ids_flat != 0
        predicted_non_padding = predicted_labels_flat[non_padding_mask]
        targets_non_padding = batch_output_ids_flat[non_padding_mask]
        correct_predictions = (predicted_non_padding == targets_non_padding).sum()

        accuracy = correct_predictions.float() / targets_non_padding.size(0) if targets_non_padding.size(0) > 0 \
            else torch.tensor([1.0])

        batch_acc = accuracy.item()
        all_predictions.append(predicted_labels_flat[non_padding_mask])
        all_targets.append(batch_output_ids_flat[non_padding_mask])
        all_masks.append(non_padding_mask)

        # print(predicted_labels, batch_output_ids)
        # loop through all predictions to get Segmentation accuracy and POS accuracy
        total = 0
        total_pos_given_cws = 0
        correct_seg = 0
        correct_pos = 0
        for i in range(len(predicted_labels)):
            tag = id_to_tag[predicted_labels[i].item()]
            targ_tag = id_to_tag[batch_output_ids[0][i].item()]
            pred = tag.split('-')
            targ = targ_tag.split('-')
            if len(pred) > 1:
                seg, pos = pred[0], pred[1]
                if len(targ) > 1:   # if false, then pred must be wrong
                    t_seg, t_pos = targ[0], targ[1]
                    if seg == t_seg:
                        correct_seg += 1
                        if pos == t_pos:    # check POS accuracy GIVEN CWS is correct
                            correct_pos += 1
                        all_pos_pred.append(pos)
                        all_pos_targ.append(t_pos)
                        total_pos_given_cws += 1
                    all_cws_pred.append(seg)
                    all_cws_targ.append(t_seg)

                total += 1
        seg_acc = correct_seg / total if correct_seg > 0 and total > 0 else batch_acc
        pos_acc = correct_pos / total_pos_given_cws if correct_pos > 0 and total_pos_given_cws > 0 else None
        total_batch += batch_acc
        total_cws += seg_acc
        total_pos = total_pos + pos_acc if pos_acc else total_pos

        print(f'batch {batch_count} / {math.ceil(len(eval_dataset) / 1)} batch accuracy {batch_acc} CWS acc {seg_acc} POS acc {pos_acc}')
        batch_count += 1
        # print(predictions)
    print(f'dev set acc {total_batch / batch_count}, dev CWS acc {total_cws / batch_count} dev POS acc {total_pos / batch_count}')
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    f1 = f1_score(all_targets.cpu().numpy(), all_predictions.cpu().numpy(), average='weighted')
    f1_cws = f1_score(all_cws_targ, all_cws_pred, average='weighted')
    f1_pos = f1_score(all_pos_targ, all_pos_pred, average='weighted')
    print(f"F1 Score Joint: {f1}, F1 Score CWS {f1_cws}, F1 Score POS {f1_pos}")

