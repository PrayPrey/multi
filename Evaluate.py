def evaluate(transformer, question, question_mask, max_len, dict):

    transformer.eval()

    bb = {v:k for k,v in tokenizer.get_vocab().items()} 
    start_token = 58101
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions[:,:-1], dim = 1)
        
        next_word = next_word.item()
  
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)


    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    print(words)

    return sentence


max_len = 128
enc_qus = [58101,    93,  2896,   564, 15202,  1729,    14, 24185,  4470,     2,
        16209,    79,    14, 27192,    18,     7, 24185,  2729,    56,     3,
            0, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100,
        58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100]
question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
question_mask = (question!=58100).to(device).unsqueeze(1).unsqueeze(1)
sentence = evaluate(transformer, question, question_mask, int(max_len), dict)
print(sentence)

# 결과값
# [58101, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
