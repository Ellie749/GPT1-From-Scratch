import torch
from process_data import train_validation_split, make_LM_dataset
from model import BiGram
from utils import plot_metrics

SEQUENCE_LENGTH = 8
EMBEDDING_DIM = 8
BATCH_SIZE = 16
EPOCHS = 10

    

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # data
    dataset = open('dataset.txt', 'r', encoding='utf-8').read()
    dictionary = ''.join(sorted(set(dataset)))
    VOCAB_SIZE = len(dictionary)
    char_id = {s:i for i, s in enumerate(dictionary)}
    id_char = {i:s for i, s in enumerate(char_id)}
    encoder = lambda s: [char_id[c] for c in s]
    decode = lambda i: ''.join([id_char[c] for c in i])
    tokenized_dataset = torch.tensor(encoder(dataset), dtype=torch.long)
    train_data, validation_data = train_validation_split(tokenized_dataset)
    X_train, y_train = make_LM_dataset(train_data, SEQUENCE_LENGTH, BATCH_SIZE)
    X_validation, y_validation = make_LM_dataset(validation_data, SEQUENCE_LENGTH, BATCH_SIZE)
    print(f"shape of x_train: {X_train.shape}")
    print(f"shape of y_train: {y_train.shape}")
    print(f"shape of x_validation: {X_validation.shape}")
    print(f"shape of y_validation: {y_validation.shape}")

    # model
    bigram_model = BiGram(VOCAB_SIZE).to(device)
    H = bigram_model.train(X_train.to(device), y_train.to(device), X_validation.to(device), y_validation.to(device), epochs=EPOCHS)
    plot_metrics(H)

    # inference
    # print(bigram_model.generate(torch.tensor([0]).to(device), 100).tolist())
    print(f"model's generated text: {decode(bigram_model.generate(torch.tensor([0]).to(device), 100).tolist())}")


if __name__=='__main__':
    main()