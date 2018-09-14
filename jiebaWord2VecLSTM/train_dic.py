from rnnforsentiment.data_process import word2vec_train, loadfile, tokenizer, readxlsx


if __name__ == '__main__':
    print("loading traning set")
    dataset = []
    for i in range(1, 7):
        dataset.append('../rawdata/train_' + str(i) + '.xlsx')
    input = readxlsx(dataset)
    print("cut the sentence")
    tokenizer(input)
    word2vec_train(input, from_begin=True)