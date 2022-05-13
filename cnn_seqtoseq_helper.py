import torch


def multi_acc(y_pred, y_test):
    """
    get the prediction accuracy
    """
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def count_parameters(model):
    """count the number of parameters in this model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    epoch_time = end_time - start_time
    epoch_mins = int(epoch_time / 60)
    epoch_seconds = int(epoch_time - (epoch_mins * 60))
    return epoch_mins, epoch_seconds

def translate_text(sentence, src_field, trg_field, model, device, max_len=100):
    """get translated text in target languagefor the test file"""
    model.eval()
    tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_conved, encoder_combined)
        predict_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(predict_token)
        if predict_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]
