import matplotlib.pyplot as plt
import dill
from math import exp
import numpy as np
from tqdm import trange
from dataset import Vocabulary

path=""

def load(path):
    with open(path, "rb") as file:
        obj = dill.load(file)
    return obj

def prepare_dataset(src_path,trg_path,build=True):
    source_language = Vocabulary(path=src_path)
    target_language = Vocabulary(path=trg_path)
    if build:
        target_language.build_vocab()
        source_language.build_vocab()

    return source_language, target_language

def visualize(sources, output='measurements.png'):
    fig = plt.figure(figsize=(12,10), dpi=180)
    ax = fig.add_subplot()
    color = ['red', 'blue', 'black', 'green']
    style = ['-', '--', ':']

    log = load("{}".format(sources[0]))
    x = np.arange(len(log))
    ax.plot(log, linewidth=1, label=sources[0])

    for i, source in enumerate(sources[1:]):
        log = load("{}".format(source))
        x = np.arange(len(log))
        ax.plot(log, linewidth=1, label=source, c=color[i % 4], linestyle=style[i // 4])

    ax.set_xlabel('Epochs',fontsize=12)
    ax.set_ylabel('Training Loss',fontsize=12)
    ax.legend()
    plt.title('Training Loss VS Epochs',fontsize=16)
    plt.savefig(output)


def BLEU_score(reference, candidate, n):
    # Calculate BLEU score for a reference (string), and a candidate(string), 
    # considering n-gram precision and brevity penalty

    def helper_grouping(seq, n):
        temp=len(seq) - n + 1
        ngrams=[seq[i:i + n] for i in range(temp)]
        return ngrams

    def helper_ngram_precision(reference, candidate, n, cap=3):
        acc=0
        dict_temp={}
        for gram in helper_grouping(candidate,n):
            if gram not in dict_temp:
                dict_temp[gram]=1
            else:
                dict_temp[gram]+=1
        for gram in helper_grouping(candidate,n):
            if gram in helper_grouping(reference,n) and dict_temp[gram]<=cap:
                acc+=1
        if len(candidate)==0 or len(helper_grouping(candidate,n))==0:
            return 0
        else: 
            return acc/len(helper_grouping(candidate,n))

    def helper_bp(reference, candidate):
        ci=len(candidate)
        ri=len(reference)
        if ci==0:
            return 0
        BP=1
        if ci<ri:
            BP*=exp(1-(ri/ci))
        return BP
        
    bleu = 1
    for i in range(1, n + 1):
        bleu *= helper_ngram_precision(reference, candidate, i)
    bleu=helper_bp(reference, candidate)*((bleu)**(1/n))

    return bleu


if __name__ == '__main__':
    #visualize(['Bi-LSTM','ConS2S_Model1', 'ConS2S_Model2', 'ConS2S_Model3'])
    pass
