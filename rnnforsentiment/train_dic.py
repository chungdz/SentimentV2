from rnnforsentiment.data_process import word2vec_train, tokenizer, readxlsx


if __name__ == '__main__':
    print("loading traning set")
    dataset = []
    for i in range(7, 8):
        dataset.append('../rawdata/train_' + str(i) + '.xlsx')
    input = readxlsx(dataset)
    print("cut the sentence")
    tokenizer(input)
    word2vec_train(input, from_begin=False)