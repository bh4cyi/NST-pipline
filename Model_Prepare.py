# -*- coding: utf-8 -*-
import vocab # Library to generate dictionary of target text.
import skipthoughts # Library to generate skip vectors.
import train # Library to generate model of target text.
import argparse
import numpy as np
import nltk
import urllib
import os.path
import sys
import re

def load_text(loc):
    text = []
    f = open(loc)
    for line in f:
        if not re.match(r'^\s*$', line):
            text.append(line.strip())
    return text


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def download_model():
    DOWNLOADS_DIR = os.getcwd()+'/skip_vector_model/'
    for url in open('skip_vector_model_url'):
        filename = url.rstrip("\n").split('/')[-1]
        filepath = os.path.join(DOWNLOADS_DIR, filename)

        if not os.path.isfile(filepath):
            print(filename + ' doesn\'t exist. Start downloading')
            urllib.urlretrieve(url, filepath, reporthook)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targe_text', help = 'Location of target text', default = './target_text/toy_story.txt')
    args = parser.parse_args()

    target_name = args.targe_text.split("/")[-1] # Get target text file name. eg. "speeches.txt"

    download_model()
    print("Loading Skip-Vector Model...")
    skmodel = skipthoughts.load_model()
    print("Done!")


    """
    Step 1: Generating dictionary for the target text.
    """
    print("Generating dictionary for the target text...")
    X = load_text(args.targe_text)
    worddict, wordcount = vocab.build_dictionary(X)
    vocab.save_dictionary(worddict, wordcount, './target_dict/%s.pkl'%target_name)
    print("Done! Saved dictionary under ./target_dict/ as %s.pkl"%target_name)

    """
    Step 2: Generating style vector for the target text.
    """
    print("Generating style vector for the target text...")
    nltk.download('punkt') # Natural Language Toolkit for skipthoughts encoder.
    print("The lenth of X is:")
    print len(X)
    skip_vector = skipthoughts.encode(skmodel, X)
    style_vector = skip_vector.mean(0) # 0 indicate that mean method is performed over multiple axes, see numpy.mean document
    np.save('./target_style/%s_style.npy'%target_name, style_vector)
    print("Done! Saved style under ./target_style/ as %s_style.npy"%target_name)

    """
    Step 3: Training model for the target text.
    """
    print("Training model for the target text...")

    C = X
    train.trainer(X, C, skmodel, './target_dict/%s.pkl'%target_name, './target_model/%s.npz'%target_name)
    print("Done! Saved model under ./target_model as %s.npz and %s.npz.pkl"%(target_name, target_name))
