from rnnforsentiment.data_process import loadfile, word2vec_load, tokenizer, get_data, batch_size
from rnnforsentiment.rnnmodel import rnnmodel, rnnModel
from keras.optimizers import Adam
import yaml
from keras.models import model_from_yaml
n_epoch = 1


def train(from_begin=False):
    # 训练模型，并保存
    print('Loading Data...')
    combined, y = loadfile(['../traindata/neg_train.xlsx'],
                           ['../traindata/pos_train.xlsx'],
                           ['../traindata/neu_train.xlsx'])
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('loading a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_load(combined)

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train = get_data(index_dict, word_vectors, combined, y)
    print("x_train.shape and y_train.shape:")
    print(x_train.shape, y_train.shape)

    model = None
    if from_begin:
        model = rnnModel(n_symbols, embedding_weights)
    else:
        with open('../parameter/lstm.yml', 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        # model = rnnModel(n_symbols, embedding_weights)
        print('loading weights......')
        model.load_weights('../parameter/lstm.h5')


    print('Compiling the Model...')
    adam = Adam(decay=1e-6, epsilon=1e-8)
    # adam = Adam(epsilon=1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print("Train...")  # batch_size=32)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)

    yaml_string = model.to_yaml()
    with open('../parameter/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../parameter/lstm.h5')



if __name__ == '__main__':
   train(from_begin=False)