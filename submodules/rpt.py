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
        input_data = [np.load(in0path), np.load(in1path)]
        answer_data = np.load(anspath)
    else:
        print("Input data is not ready. Try \"python PokerAI --rptg\"")
        return
    train(input_data, answer_data)

def train(input_data, answer_data):
    model = None

    with "data/rp_model" as inpath:
        if os.path.isfile(inpath):
            print("Previous trained model exists.")
            c = ''
            while c not in ['c', 'C', 'o', 'O']:
                c = input("Continue or Override? (c/o)")
                if c == 'c' or c == 'C': model = load_model(inpath)
    
    if model == None:
        cards = [Input(shape = (54,)) for i in range(2)]

        cc = layers.Concatenate()(cards)
        for i in range(3): cc = layers.Dense(108, activation = "relu")(cc)
        fin = layers.Concatenate()([cc, *cards])
        for i in range(2): fin = layers.Dense(162, activation = "relu")
        fin = layers.Dense(10, activation = "softmax")

        model = Model(cards, fin)
        model.compile(optimizer = "nadam", loss = "mean_squared_error", metrics = ['acc'])

    model.fit([input_data[0], input_data[1]], answer_data, epochs = 10000, batch_size = 8)

if __name__ == "__main__": main()