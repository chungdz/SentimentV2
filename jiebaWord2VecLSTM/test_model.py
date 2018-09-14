from rnnforsentiment.data_process import loadfile, word2vec_load, tokenizer, get_data, batch_size
import yaml
from keras.models import model_from_yaml


if __name__ == '__main__':
    print('Loading Data...')
    combined, y = loadfile(['../testdata/neg_test.xlsx'], ['../testdata/pos_test.xlsx'], ['../testdata/neu_test.xlsx'])
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_load(combined)

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_test, y_test, = get_data(index_dict, word_vectors, combined, y)
    print("x_train.shape and y_train.shape:")
    print(x_test.shape, y_test.shape)
    print('loading model......')

    with open('../parameter/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../parameter/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)