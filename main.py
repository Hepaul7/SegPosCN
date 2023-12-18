from training import *

#####
# This file is for testing (for now)
#####
print('running CTB7...\n')
data = read_csv('data/CTB7/train.tsv')

texts, tags, max_len = extract_sentences(data)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

input_ids, attention_masks = prepare(texts, bert_tokenizer, 64)  # includes BOS/EOS
output_ids, output_masks = prepare_outputs(tags, 64)  # includes BOS/EOS
bert_model = load_model('bert-base-chinese')

# input_embeddings = get_bert_embeddings(model, input_ids, attention_masks)
# output_embeddings = get_bert_embeddings(model, decoder_input_ids, decoder_attention_mask)
encoder = make_encoder()
decoder = make_decoder()
segpos_model = CWSPOSTransformer(encoder, decoder, output_size=33 * 4 + 2, d_model=768)

# change to embeddings later
dataset = TensorDataset(input_ids, attention_masks, output_ids, output_masks)
dataloader = DataLoader(dataset, batch_size=1000000, shuffle=False)

num_epochs = 100
optimizer = AdamW(segpos_model.parameters(), lr=5e-5)
loss_function = CrossEntropyLoss()

# load eval set
eval_set = read_csv('data/CTB7/dev.tsv')

eval_texts, eval_tags, _ = extract_sentences(eval_set)
eval_input_ids, eval_attention_masks = prepare(eval_texts, bert_tokenizer, 64)
eval_output_ids, eval_output_masks = prepare_outputs(eval_tags, 64)
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_output_ids, eval_output_masks)
eval_dataloader = DataLoader(eval_dataset, batch_size=40000, shuffle=False)


for epoch in range(num_epochs):
    avg_train_loss = train_epoch(segpos_model, bert_model, dataloader, loss_function, optimizer)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss}")

    # Evaluate after each epoch
    avg_eval_loss, accuracy, f1 = evaluate_model(segpos_model, bert_model, eval_dataloader, loss_function)
    print(f"Epoch {epoch+1} Evaluation Metrics:\n Loss: {avg_eval_loss}, Accuracy: {accuracy}, F1 Score: {f1}")


