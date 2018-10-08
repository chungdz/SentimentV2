import numpy as np
import multiprocessing
from gensim.models.word2vec import Word2Vec
import gensim
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import jieba
import keras
import xlrd

batch_size = 32
cpu_count = multiprocessing.cpu_count()
n_iterations = 3
window_size = 7
vocab_dim = 70
n_exposures = 5
maxlen = 100

def readxlsx(filenames, string_col=0):
    string_file = []

    for filename in filenames:
        file = xlrd.open_workbook(filename)
        for sheet in file.sheets():
            for i in range(sheet.nrows):
                string_file.append(str(sheet.cell(i, string_col).value))

    print("一共" + str(len(string_file)) + "行")
    return string_file


def loadfile(negUrls, posUrls, neuUrls):
    # 返回长度等于csv行数的数组
    # neg = pd.read_csv(negUrl, header=None, index_col=None)
    # pos = pd.read_csv(posUrl, header=None, index_col=None, error_bad_lines=False)
    # neu = pd.read_csv(neuUrl, header=None, index_col=None)
    neg_raw = readxlsx(negUrls)
    pos_raw = readxlsx(posUrls)
    neu_raw = readxlsx(neuUrls)

    neg = [neg_raw]
    pos = [pos_raw]
    neu = [neu_raw]
    #返回三个数组的拼接numpy数组
    combined = np.concatenate((pos[0], neu[0], neg[0]))
    #返回长度和combined相同，但是值对应于情感正负性的y数组
    y = np.concatenate((np.ones(len(pos_raw), dtype=int), np.zeros(len(neu_raw), dtype=int),
                        2 * np.ones(len(neg_raw), dtype=int)))

    return combined, y


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '').replace('0xb0', '')) for document in text]
    return text


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined, from_begin=False):
    model = None
    if from_begin:
        model = Word2Vec(size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         iter=n_iterations,
                         sg=1)
        model.build_vocab(combined)
        model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    else:
        model = gensim.models.Word2Vec.load('../parameter/Word2vec_model.pkl')
        model.build_vocab(combined, update=True)
        model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('../parameter/Word2vec_model.pkl')
    print("saved")


def word2vec_load(combined):
    model = gensim.models.Word2Vec.load('../parameter/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def create_dictionaries(model=None, combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #index 0 为不知道的词语， 所以所有index加1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        #单词到训练好的vector
        w2vec = {word: model[word] for word in w2indx.keys()}
        #将输入的句子每一个单词转换成index
        combined = parse_dataset(combined, w2indx)
        #一个句子最多maxlen个单词，不够的在句子前面补0, 多出的截断前面的句子
        combined = sequence.pad_sequences(combined, maxlen=maxlen, padding='pre', value=0.0, truncating='pre')#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#word 2 index
def parse_dataset(combined, w2idict):
    ''' Words become integers
    '''
    data = []
    for sentence in combined:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2idict[word])
            except:
                new_txt.append(0) # freqxiao10->0
        data.append(new_txt)
    return data


def get_data(index_dict, word_vectors, combined, y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于n_exposures的词语索引为0，所以加1
    # 给embedding层字段矩阵
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    y = keras.utils.to_categorical(y, num_classes=3)
    # y_test = keras.utils.to_categorical(y_test, num_classes=3)
    print(combined.shape, y.shape)
    return n_symbols, embedding_weights, combined, y