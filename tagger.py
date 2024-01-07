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
        output_embeddings = torch.cat([enc_output[:, :trg_output.shape[1], :].clone(), output_embeddings.clone()], dim=-1)
        output_embeddings = self.model.dim_alignment(output_embeddings.clone())

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
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask, trg_mask)
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
        # print('genseq shape ans_idx', gen_seq[ans_idx].shape)
        return gen_seq[ans_idx]
        # return gen_seq[ans_idx][:seq_lens[ans_idx]]
