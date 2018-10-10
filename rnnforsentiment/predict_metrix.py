#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
from rnnforsentiment.data_process import word2vec_load, tokenizer, get_data, readxlsx
import yaml
from keras.models import model_from_yaml
import numpy as np
import xlsxwriter

batch_size = 32

def predictSentiment(filename, resultname):
    # string_file = pd.read_csv(filename, header=None, index_col=None)
    # reader = csv.reader(open(filename, 'r', encoding='utf-8'))
    # string_file = []
    # for v in reader:
    #     string_file.append(v[0])
    string_file = readxlsx([filename])

    input = tokenizer(string_file)
    index_dict, word_vectors, input = word2vec_load(input)

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_test, _, = get_data(index_dict, word_vectors, input, ['1'])

    with open('../parameter/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../parameter/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predicted = model.predict(x_test, batch_size=batch_size)
    print("predicted shape: " + str(predicted.shape))
    result = np.argmax(predicted, axis=1)
    # result = predicted
    newfile = xlsxwriter.Workbook(resultname)
    sheet = newfile.add_worksheet('sheet1')
    row = 0
    for v in result:
        sheet.write(row, 1, string_file[row])

        if v == 0:
            sheet.write(row, 0, '中')
        elif v == 1:
            sheet.write(row, 0, '正')
        elif v == 2:
            sheet.write(row, 0, '负')
        row += 1
    newfile.close()


if __name__ == '__main__':
    # predictSentiment('../predictdata/predicted_pos.xlsx', '../predictdata/result_pos.xlsx')
    # predictSentiment('../predictdata/predicted_neg.xlsx', '../predictdata/result_neg.xlsx')
    # predictSentiment('../predictdata/predicted_neu.xlsx', '../predictdata/result_neu.xlsx')
    predictSentiment('../predictdata/sample.xlsx', '../predictdata/result.xlsx')
    # predictSentiment('../predictdata/random.xlsx', '../predictdata/result_ran.xlsx')


