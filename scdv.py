import gensim
import pandas as pd
import numpy as np 
import scipy as sp
from collections import defaultdict
from sklearn.mixture import GaussianMixture

class SCDV(object):
    # モデルを使って言葉のベクトルを取得 .binバージョン
    def get_word_vector(self, words, vec_size, model_path):
        bin_flag = False
        if 'bin' in model_path[-3:]:
            bin_flag = True
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=bin_flag)

        word_vec = defaultdict(list)
        for word in words:
            try:
                word_vec[word].append(model[word])
            except:
                word_vec[word].append(np.zeros(vec_size, ))

        return word_vec

    # 出現頻度の数え上げ(1単語は1データに1回)
    def count_freq(self, words, sep=','):
        frequency = defaultdict(int)
        for i, word in enumerate(words):
            word_list = []
            for token in word.split(sep):
                if token in word_list: continue
                frequency[token] += 1
                word_list.append(token)
        return frequency

    # IDFの計算
    def calc_idf(self, n, word_freq):
        return np.log2(n/word_freq) + 1.0

    # GMM
    def training_GMM(self, wv, n_cluster, vec_size):
        x = [vec[0].reshape(1, vec_size)[0] for word, vec in wv.items() if not np.all(vec[0] == 0)]

        gmm = GaussianMixture(n_components=n_cluster, covariance_type='tied')
        gmm.fit(np.array(x))
        return gmm

    def calc_scdv(self, words, gmm, n, wv, freq, cluster=5, vec_size=200, sep=','):
        # SCDV計算
        sentence_vec = []
        for row in words:
            wtv = np.zeros(cluster*vec_size, )
            for word in row.split(sep):
                if np.all(wv[word] == 0): continue
                # idf
                idf = self.calc_idf(n, freq[word])
                # wcv_ik
                wcv_ik = [prob * wv[word][0] for prob in gmm.predict_proba(wv[word])[0]]
                # wtv_i
                con = np.concatenate((wcv_ik[0], wcv_ik[1]))
                if len(wcv_ik) > 2:
                    for wcv in wcv_ik[2:]:
                        con = np.concatenate((con, wcv))
                wtv_i = con * idf
                wtv += wtv_i
            sentence_vec.append(wtv/len(row.split(sep)))
        return sentence_vec

if __name__ == '__main__':
    scdv = SCDV()
    
    # data
    path = './dogura_magura.csv'
    scdv_column = 'parsed'
    words = pd.read_csv(path)[scdv_column].values

    # 単語頻度
    freq = scdv.count_freq(words)
    N = len(words)

    # modelファイルの読み込み
    model_path = './entity_vector/entity_vector.model.bin'

    # modelのベクトルサイズ
    vec_size = model.vector_size
    print('vector length:{0}'.format(vec_size))

    # word vectorを取得
    wv = scdv.get_word_vector(freq.keys(), vec_size, model)

    # モデルの削除
    del model
    
    # GMM
    cluster = 5
    gmm = scdv.training_GMM(wv, n_cluster=cluster, vec_size=vec_size)
    
    #scdv
    sentence_vectors = scdv.calc_scdv(words, gmm, N, wv, freq, cluster, vec_size, sep=',')