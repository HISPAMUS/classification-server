import numpy as np
import logging
import datetime
from modelTemplates.kerasModel import KerasModel


__all__ = [ 'SymbolClassifier' ]


class SymbolClassifier:

    logger = logging.getLogger('SymbolClassifier')
    lastUsed = None


    def __init__(self, model_shape_path, model_position_path, vocabulary_shape, vocabulary_position):
        self.logger.info('Loading models...')

        self.shapeClassifier = KerasModel(model_shape_path)

        shape_vocabulary = np.load(vocabulary_shape, allow_pickle=True).item()  # Category -> int
        self.shape_vocabulary = dict((v, k) for k, v in shape_vocabulary.items())  # int -> Category

        self.positionClassifier = KerasModel(model_position_path)

        position_vocabulary = np.load(vocabulary_position, allow_pickle=True).item()  # Category -> int
        self.position_vocabulary = dict((v, k) for k, v in position_vocabulary.items())  # int -> Category

        self.logger.info('Models loaded')

        self.lastUsed = datetime.datetime.now()


    def predict(self, shape_image, position_image, n):
        
        self.lastUsed = datetime.datetime.now()
        # Predictions
        shape_prediction_all = self.shapeClassifier.predict(shape_image)
        #self.logger.info(shape_prediction_all)
        #shape_prediction = np.argmax(shape_prediction_all)
        #self.logger.info(shape_prediction)
        shape_prediction = np.flip(np.argsort(shape_prediction_all.flatten()))[0:n] # Equivalent to argmax returning the index of the n maxmimum values
        #self.logger.info(shape_prediction)

        position_prediction_all = self.positionClassifier.predict(position_image)
        #self.logger.info(position_prediction_all)
        #position_prediction = np.argmax(position_prediction_all)
        #self.logger.info(position_prediction)
        position_prediction = np.flip(np.argsort(position_prediction_all.flatten()))[0:n]
        #self.logger.info(position_prediction)

        return ([self.shape_vocabulary[x] for x in shape_prediction], [self.position_vocabulary[x] for x in position_prediction])
    
    def getLastUsed(self):
        
        return self.lastUsed
