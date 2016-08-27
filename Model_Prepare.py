import vocab # Library to generate dictionary of target text.
import skipthoughts # Library to generate skip vectors.
import train # Library to generate model of target text.
import argparse
import numpy as np

def load_text(loc):
    text = []
    with open(loc, 'rb') as f:
        for line in f:
            text.append(line.strip())
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targe_text', help = 'Location of target text', default = './target_text/speeches.txt')
    args = parser.parse_args()

    target_name = args.targe_text.split("/")[-1] # Get target text file name. eg. "speeches.txt"

    """
    Step 1: Generating dictionary for the target text.
    """
    print("Generating dictionary for the target text...")
    X = load_text(args.targe_text)
    worddict, wordcount = vocab.build_dictionary(X)
    vocab.save_dictionary(worddict, wordcount, './target_dict/%s.pkl'%target_name)
    print("Done! Saved dictionary under ./target_dict/ as %s.pkl"%target_name)

    """
    Step 2: Training model for the target text.
    """
    print("Training model for the target text...")
    skmodel = skipthoughts.load_model()

    C = X
    train.trainer(X, C, skmodel, s'./target_dict/%s.pkl'%target_name, './target_model/%s.npz'%target_name)
    print("Done! Saved model under ./target_model as %s.npz and %s.npz.pkl"%(target_name, target_name))

    """
    Step 3: Generating style vector for the target text.
    """
    print("Generating style vector for the target text...")
    nltk.download('punkt') # Natural Language Toolkit for skipthoughts encoder.
    skip_vector = skipthoughts.encode(skmodel, X)
    style_vector = skip_vector.mean(0) # 0 indicate that mean method is performed over multiple axes, see numpy.mean document
    np.save('./target_style/%s_style.npy'%target_name, style_vector)
    print("Done! Saved style under ./target_style/ as %s_style.npy"%target_name)
