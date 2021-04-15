import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
import itertools

import logging
from logger import Logger

class Seq2Seq_Translator_K:
    
    def __init__(self, *args, **kwargs):
        model_path = kwargs["model_path"]
        self.i2w = np.load(kwargs["semantic_i2w"], allow_pickle=True).item()
        self.w2i_agnostic = np.load(kwargs["agnostic_w2i"], allow_pickle=True).item()
        self.w2i_semantic = np.load(kwargs["semantic_w2i"], allow_pickle=True).item()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        K.clear_session()

        if K.backend() == 'tensorflow':
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            self.session = sess
        
        self.model = load_model(model_path)
        self.model._make_predict_function()


    def predict(self, *args, **kwargs):
        #print(kwargs['input'].split('\t'))
        input_sequence = [self.w2i_agnostic[token.strip()] if token.strip() in self.w2i_agnostic else self.w2i_agnostic['<unk>'] for token in kwargs['input'].split(',')]
        print(input_sequence)
        decoded = [self.w2i_semantic['<sos>']]
        predicted = []
        for i in range(1, 100):
            decoder_input = np.asarray([decoded])
            prediction = self.model.predict([np.asarray([input_sequence]).astype('float32'), decoder_input])
            #decoded.append(0)
            decoded.append(np.argmax(prediction[0][-1]))
            if self.i2w[np.argmax(prediction[0][-1])] == '<eos>':
                break
        
            predicted.append(self.i2w[np.argmax(prediction[0][-1])])
        
        prediction = ""
        for token in predicted:
            prediction+=token
            prediction += '\n'

        return prediction
    
    def close(self):
        self.session.close()





