from embeddings import *
from encodings import *

data = read_csv('data/CTB7/dev.tsv')
texts = extract_sentences(data)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
input_ids, attention_masks = prepare(texts, bert_tokenizer)

model = load_model('bert-base-chinese')
embeddings = get_bert_embeddings(model, input_ids, attention_masks)

encoder = Encoder()
encoder_output = encoder()