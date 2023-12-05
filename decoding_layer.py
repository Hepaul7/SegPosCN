# import torch.nn as nn
# from transformers import BertModel
# from TorchCRF import CRF
#
#
# class CRFDecoder(nn.Module):
#     def __init__(self, bert_model_name, num_labels):
#         super(CRFDecoder, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#         self.crf = CRF(num_labels)
#
#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         sequence_output = self.dropout(outputs.last_hidden_state)
#         logits = self.classifier(sequence_output)
#
#         if labels is not None:
#             loss = -self.crf(logits, labels, mask=attention_mask.byte())
#             return loss
#         else:
#             return self.crf.decode(logits, mask=attention_mask.byte())
