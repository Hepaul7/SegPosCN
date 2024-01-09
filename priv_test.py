from transformer.encoder import *
from transformer.decoder import *
from models import *
from tagger import Tagger

OUTPUT_SIZE = 33 * 4 + 5

data = read_csv('data/CTB9/dev.tsv')

texts, tags, max_len = extract_sentences(data)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
output_ids, output_masks = prepare_outputs(tags, max_len=64)  # includes BOS/EOS
model = load_model('bert-base-chinese')
model.load_state_dict(torch.load('bert_state_dict9-2'))

encoder = make_encoder()
decoder = make_decoder()
output_embedder = OutputEmbedder(768, 768)
positional_encoder = PositionalEncoder(768, drop_out=0.1, max_len=64)
segpos_model = CWSPOSTransformer(encoder, decoder, output_size=OUTPUT_SIZE, d_model=768,
                                 output_embedder=output_embedder, positional_encoder=positional_encoder)

print('loading state')
segpos_model.load_state_dict(torch.load(f'model_state_dict.pth'))
segpos_model.eval()

with torch.no_grad():
    txt = input('enter sentence: ')
    my_text = [txt]
    input_ids, mask = prepare(my_text, bert_tokenizer, 64)
    input_embeddings = get_bert_embeddings(model, input_ids, mask)
    tagger = Tagger(segpos_model, 4, 64, 0, 0, 133, 134)
    predicted_labels = tagger.generate(input_embeddings, mask, mask)

    # print(predicted_labels)

    predicted_tags = []
    for pred in predicted_labels:
        if id_to_tag[pred.item()] != '[PAD]':
            predicted_tags.append(id_to_tag[pred.item()])
    print('your text: ', my_text)
    print('predicted tags with CLS, SEP: ', predicted_tags)
