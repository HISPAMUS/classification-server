import numpy as np
from keras.models import load_model
from keras import backend as K

class KerasModel:

    def __init__(self, model_path):
        K.set_image_data_format("channels_last")
        if K.backend() == 'tensorflow':
            import tensorflow as tf    # Memory control with Tensorflow
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
        
        self.model = load_model(model_path)
        self.model._make_predict_function()

    
    def getModel(self):
        return self.model