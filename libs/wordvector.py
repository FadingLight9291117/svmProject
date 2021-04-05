import multiprocessing
import functools

from gensim.models.word2vec import Word2Vec
import jieba
import numpy as np
import pandas as pd


def build_word_vec_model(words, vector_size):
    """
        构建向量模型
    """
    w2v_model = Word2Vec(min_count=0,  # 词频低于min_count的此会被删除
                         vector_size=vector_size,  # 设置词向量的维度
                         workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(words)
    w2v_model.train(words, epochs=10, total_examples=w2v_model.corpus_total_words)
    return w2v_model


def build_sentence_vec(sentence, vector_size, w2v_model):
    """
        构建句向量
    :param sentence:
    :param vector_size:
    :param w2v_model:
    """
    vec = np.zeros(vector_size)
    words = jieba.cut(sentence)
    for word in words:
        try:
            _vec = w2v_model.wv.get_vector(word)
            vec += _vec
        except KeyError:
            continue

    return vec


def sentence_vec():
    print('load w2v model begin')
    w2v_model = Word2Vec.load('./models/word_vec_model.model')
    print('load w2v model end')
    vector_size = w2v_model.wv.vector_size
    return functools.partial(build_sentence_vec, vector_size=vector_size, w2v_model=w2v_model)


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    raw_data = df['text'].to_numpy()
    words = []
    for sentence in raw_data:
        _words = jieba.cut(sentence)
        _words = list(_words)
        words.extend(_words)
    model = build_word_vec_model(words, 100)
    model.save('./models/word_vec_model.model')
