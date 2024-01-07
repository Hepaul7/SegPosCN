import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CWSPOSTransformer
from transformer.decoder import get_subsequent_mask


class Tagger(nn.Module):
    """
    Load a trained CWSPOSTransformer model and tag in beam search fashion.
    Based on https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py
    """

    def __init__(self, model: CWSPOSTransformer, beam_size, max_seq_len, src_pad_idx, trg_pad_idx, trg_bos_idx,
                 trg_eos_idx):
        super(Tagger, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        # self.blank_seqs[:, 1] = 135
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_output, enc_output, src_mask, trg_mask):  # trg_mask should be removed
        output_embeddings = self.model.tgt_embedder(trg_output)
        output_embeddings = self.model.position(output_embeddings)  # add PE
        trg_mask = torch.tensor([[1] * trg_output.shape[1]])
        dec_output = self.model.decoder(output_embeddings, enc_output, src_mask, trg_mask.float())
        return F.softmax(self.model.output(dec_output), dim=2)

    def _get_init_state(self, src_embeddings, src_mask, trg_mask):
        beam_size = self.beam_size

        # enc_output = self.model.encoder(src_embeddings, src_mask.float())

        # print(enc_output.shape)
        init_seq = self.init_seq
        dec_output = self._model_decode(init_seq, src_embeddings, src_mask, trg_mask)
        # print(dec_output.shape)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)
        # print(best_k_probs, best_k_idx)

        scores = best_k_probs.view(beam_size)  # I removed torch.log
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = src_embeddings.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        beam_size = self.beam_size

        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        # print(dec_output[:, -1, :].shape) -> torch.Size([4, 137]) seems correct
        # print(best_k2_probs.shape, best_k2_idx.shape) -> torch.Size([4, 4]) torch.Size([4, 4])
        scores = best_k2_probs.view(beam_size, -1) + scores.view(beam_size, 1)
        # print(scores.shape)  -> torch.Size([4, 4]) seems correct
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx
        return gen_seq, scores

    def generate(self, src_embeddings, src_mask, trg_mask):
        # as the original code, only batch_size = 1
        assert src_embeddings.size(0) == 1
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha
        with torch.no_grad():
            enc_output, gen_seq, scores = self._get_init_state(src_embeddings, src_mask, trg_mask)
            ans_idx = 0
            for step in range(2, max_seq_len):
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask, trg_mask[:, :step])
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        print('genseq shape ans_idx', gen_seq[ans_idx].shape)
        return gen_seq[ans_idx]
        # return gen_seq[ans_idx][:seq_lens[ans_idx]]


"""BELOW FOR TESTING PURPOSES ONLY"""

# from torch.utils.data import TensorDataset, DataLoader
# from transformer.encoder import *
# from transformer.decoder import *
# from models import *
#
# OUTPUT_SIZE = 33 * 4 + 5
#
# data = read_csv('data/CTB7/train.tsv')
#
# texts, tags, max_len = extract_sentences(data)
#
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#
# input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
# output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
# model = load_model('bert-base-chinese')
# for param in model.parameters():
#     param.requires_grad = False
#
# encoder = make_encoder()
# decoder = make_decoder()
# output_embedder = OutputEmbedder(768, 768)
# positional_encoder = PositionalEncoder(768, drop_out=0.1, max_len=64)
# segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768,
#                                  output_embedder=output_embedder, positional_encoder=positional_encoder)
#
# print('loading state')
# segpos_model.load_state_dict(torch.load('model_state_dict.pth'))
# segpos_model.eval()
# tagger = Tagger(segpos_model, 4, 64, 0, 0, 133, 134)
#
# eval_set = read_csv('data/CTB7/train.tsv')
# #
# eval_texts, eval_tags, _ = extract_sentences(eval_set)
# # print(eval_texts)
# eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
# eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
# eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
# eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
# all_predictions = []
# all_targets = []
# all_masks = []
# batch_count = 1
#
# with torch.no_grad():  # No gradients needed for evaluation
#     for batch in eval_dataloader:
#         batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
#         input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)
#
#         # tagger._get_init_state(input_embeddings, batch_attention_masks, batch_output_masks)
#         # predictions = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)
#         predicted_labels = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)
#
#         # loss = loss_function(predictions.view(-1, predictions.size(-1)).float(), batch_output_ids.view(-1).long())
#         # predicted_labels = torch.argmax(predictions, dim=2)
#         batch_output_ids = batch_output_ids
#         predicted_labels = predicted_labels.clone() * batch_attention_masks
#         batch_output_ids = batch_output_ids.clone() * batch_output_masks
#         predicted_labels = predicted_labels.clone().float()
#         batch_output_ids = batch_output_ids.clone().float()
#         predicted_labels.requires_grad = False
#         # total_eval_loss += loss.item()
#
#         predicted_labels_flat = predicted_labels.view(-1)
#         batch_output_ids_flat = batch_output_ids.view(-1)
#         non_padding_mask = batch_output_ids_flat != 0
#         predicted_non_padding = predicted_labels_flat[non_padding_mask]
#         targets_non_padding = batch_output_ids_flat[non_padding_mask]
#         correct_predictions = (predicted_non_padding == targets_non_padding).sum()
#         accuracy = correct_predictions.float() / targets_non_padding.size(0)
#
#         batch_acc = accuracy.item()
#         all_predictions.append(predicted_labels_flat[non_padding_mask])
#
#         all_targets.append(batch_output_ids_flat[non_padding_mask])
#         all_masks.append(non_padding_mask)
#
#         print(predicted_labels, batch_output_ids)
#         print(f'batch {batch_count} / {math.ceil(len(eval_dataset) / 1)} batch accuracy {batch_acc}')
#         batch_count += 1
        # print(predictions)
        # print(batch_output_ids)
