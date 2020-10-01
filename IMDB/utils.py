from torchtext import data
from torchtext import datasets
import torch

def get_data(bs,embedding_size):
    text = data.Field(tokenize='spacy', include_lengths=True)
    label = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(text, label)
    train_data, valid_data = train_data.split()

    max_vocab_size = 25_000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B."+str(embedding_size)+"d",
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=bs,sort=False)

    return train_iterator, valid_iterator, test_iterator, text

def zero_words_in_embedding(model, embedding_size, text, pad_idx):
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = text.vocab.stoi[text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_size)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_size)