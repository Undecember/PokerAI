import numpy as np
import os
from keras.models import Model, Sequential, load_model
from keras import layers
from keras import Input

def main():
    in0path = "data/rp_input_data0.npy"
    in1path = "data/rp_input_data1.npy"
    anspath = "data/rp_answer_data.npy"
    if False not in [
        os.path.isfile(in0path),
        os.path.isfile(in0path),
        os.path.isfile(in0path)]:
        print("Previous trained model exists.")
        c = ''
        while c not in ['c', 'C', 'o', 'O']:
            c = input("Continue or Override? (c/o)")
            if c == 'c' or c == 'C': model = load_model(inpath)
        input_data = [np.load(in0path), np.load(in1path)]
        answer_data = np.load(anspath)
    else:
        print("Input data is not ready. Try \"python PokerAI --rptg\"")
        return
    train(input_data, answer_data)

if __name__ == "__main__": main()