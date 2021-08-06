from __future__ import print_function
import IPython
import sys

import keras.models
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import pygame

IPython.display.Audio('./data/30s_seq.mp3')
# pygame.mixer.init()
# track=pygame.mixer.music.load("./data/30s_seq.mp3")
# pygame.mixer.music.play()

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

n_a = 64

reshapor = Reshape((1, 90))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D

def djmodel(Tx,n_a,n_values):
    outputs=[]
    X=Input(shape=(Tx,n_values))
    a0=Input((n_a,),name='a0')
    c0=Input((n_a,),name="c0")
    a=a0#这里不用a0的话，下面建立Model时会认为模型的输出是最近得到的a,c，而非最开始的a0，c0
    c=c0
    for t in range(Tx):
        x=Lambda(lambda x:x[:,t,:])(X)
        x=reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out=densor(a)
        outputs.append(out)
    model=Model(inputs=[X,a0,c0],outputs=outputs)
    return model


opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, decay=0.01)
model=djmodel(X.shape[1],n_a,n_values)
model.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
m=60
a0=np.zeros((m,n_a))
c0=np.zeros((m,n_a))
model.fit([X,a0,c0],list(Y),epochs=100)


def music_inference_model(LSTM_cell,densor,n_values=90,n_a=64,Ty=100):
    outputs=[]
    x0 = Input(shape=(1,n_values))
    a0 = Input((n_a,), name='a0')
    c0 = Input((n_a,), name="c0")
    a = a0  # 这里不用a0的话，下面建立Model时会认为模型的输出是最近得到的a,c，而非最开始的a0，c0
    c = c0
    x=x0
    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model


inference_model = music_inference_model(LSTM_cell, densor, n_values = 90, n_a = 64, Ty = 50)#用参数LSTM_cell保存训练好的模型

x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer,
                       c_initializer = c_initializer):
    pred=inference_model.predict([x_initializer,a_initializer,c_initializer])
    indices=np.argmax(pred,-1)
    result=to_categorical(indices,num_classes=x_initializer.shape[-1])
    return result,indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print(results.shape)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

out_stream = generate_music(inference_model)