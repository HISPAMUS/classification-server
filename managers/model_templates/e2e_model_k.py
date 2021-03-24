import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
import itertools

import logging
from logger import Logger

logger_term = Logger()

class E2E_K:
   
    def __init__(self, *args, **kwargs):
        try:     
            model_path = kwargs["model_path"]
            self.i2w = np.load(kwargs["w2i"], allow_pickle=True).item()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.reset_default_graph()
            K.clear_session()

            self.w2i = dict((v,k) for k, v in self.i2w.items())
            #logger_term.LogInfo(self.i2w)
            
            K.set_image_data_format("channels_last")
            if K.backend() == 'tensorflow':
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                sess = tf.Session(config=config)
                self.session = sess
            
            self.model = load_model(model_path)
            self.model._make_predict_function()
            self.HEIGHT = 64
            self.WIDTH_REDUCTION = 2
        
        except Exception as e:
            logger_term.LogError(e)
        
        
    def predict(self, *args, **kwargs):
        #original_image = cv2.imread(image_path,True)
        try:
            image = kwargs["image"]
            original_image_shape = image.shape

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Pre-process
            new_width = int(float(self.HEIGHT * image.shape[1]) / image.shape[0])
            image = cv2.resize(image, (new_width, self.HEIGHT))

            image = (255.-image)/255
            image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

            #seq_lengths = [ image.shape[2] / self.WIDTH_REDUCTION ]

            prediction = self.model.predict(image)[0]
            out_best = np.argmax(prediction,axis=1)

            # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
            #out_best = [k for k, g in itertools.groupby(list(out_best))]
            decoded = []
            for c in out_best:
                if c < len(self.i2w):  # CTC Blank must be ignored
                    if '<sos>' in self.w2i or '<eos>' in self.w2i:
                        if c != self.w2i['<sos>'] and c != self.w2i['<eos>']:
                            decoded.append(c)
                    else:
                        decoded.append(c)
                else:
                    decoded.append(-1)
            
            #logger_term.LogInfo(decoded)

            result = []
            width_refactor = 1.
            width_refactor *= original_image_shape[0]
            width_refactor /= self.HEIGHT
            width_refactor *= self.WIDTH_REDUCTION
            
            prev = -1
            for idx, w in enumerate(decoded):
                if w != '<blank>':
                    if prev == -1:
                        start = idx
                    elif prev != w:
                        if prev != -1:
                            result.append((self.i2w[prev], int(start*width_refactor), int(idx*width_refactor)))
                        start = idx
                else:
                    if prev != -1 and prev != -1:
                        result.append((self.i2w[prev], int(start*width_refactor), int(idx*width_refactor)))
                prev = w

            return result
        except Exception as e:
            logger_term.LogError(e)
	
    
    def close(self, *args, **kwargs):
        self.session.close()
