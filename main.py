from transformer.embedding_layer import *
from transformer.encoder import make_encoder
from transformer.decoder import make_decoder

print('running CTB7...\n')
data = read_csv('data/CTB7/dev.tsv')

texts, max_len = extract_sentences(data)
# labels, max_len_2 = extract_labels(data)

# assert max_len == max_len_2

# prep_text = prepare_data(texts, max_len)
# prep_label = prepare_data(labels, max_len)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# data_tensor = convert_to_tensor(texts, max_len)
input_ids, attention_masks = prepare(texts, bert_tokenizer, max_len)


model = load_model('bert-base-chinese')
# device = torch.device('mps')
# model.to(device)
# get_bert_embeddings(model, input_ids, attention_masks)
input_embeddings = get_bert_embeddings(model, input_ids, attention_masks)

# input_ids, attention_masks = prepare(labels, bert_tokenizer)
# #
# # encoder = Encoder()
# # encoder_output = encoder()
#
#

# Test Run to make sure bug free
# sample sentence


