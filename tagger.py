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

        # self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_output, enc_output, src_mask, trg_mask):
        output_embeddings = self.model.tgt_embedder(trg_output)
        output_embeddings = self.model.position(output_embeddings)
        dec_output = self.model.decoder(output_embeddings, enc_output, src_mask, trg_mask.float())
        return F.softmax(self.model.output(dec_output), dim=1)

    def _get_init_state(self, src_embeddings, src_mask, trg_mask):
        beam_size = self.beam_size

        enc_output = self.model.encoder(src_embeddings, src_mask.float())
        # I don't know if this is okay, because I pass in 0 for all unknown vals
        # which is the PAD index, but maybe I can mutate the tensor?
        init_seq = torch.zeros(enc_output.shape[0], enc_output.shape[1], dtype=torch.long)
        init_seq[:, 0] = self.trg_bos_idx
        dec_output = self._model_decode(init_seq, enc_output, src_mask, trg_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        beam_size = self.beam_size
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
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
                print(step)
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
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


"""BELOW FOR TESTING PURPOSES ONLY"""

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
tagger = Tagger(segpos_model, 64, 64, 0, 0, 133, 134)

eval_set = read_csv('data/CTB7/dev.tsv')
#
eval_texts, eval_tags, _ = extract_sentences(eval_set)
# print(eval_texts)
eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
with torch.no_grad():  # No gradients needed for evaluation
    for batch in eval_dataloader:
        batch_input_ids, batch_attention_masks, batch_output_ids, batch_output_masks = batch
        input_embeddings = get_bert_embeddings(model, batch_input_ids, batch_attention_masks)

        # tagger._get_init_state(input_embeddings, batch_attention_masks, batch_output_masks)
        predictions = tagger.generate(input_embeddings, batch_attention_masks, batch_output_masks)
        print(predictions)
        print(batch_output_ids)
