import numpy as np
from keras.models import load_model

class SymbolsModel:

    def __init__(self, model_shape_path, model_position_path, vocabulary_shape, vocabulary_position):
        self.shapeClassifier = load_model(model_shape_path)
        self.shapeClassifier._make_predict_function()

        shape_vocabulary = np.load(vocabulary_shape, allow_pickle=True).item()  # Category -> int
        self.shape_vocabulary = dict((v, k) for k, v in shape_vocabulary.items())  # int -> Category

        self.positionClassifier = load_model(model_position_path)

        position_vocabulary = np.load(vocabulary_position, allow_pickle=True).item()  # Category -> int
        self.position_vocabulary = dict((v, k) for k, v in position_vocabulary.items())  # int -> Category

    def predict(self, shape_image, position_image, n):
        shape_prediction_all = self.shapeClassifier.predict(shape_image)

        shape_prediction = np.flip(np.argsort(shape_prediction_all.flatten()))[0:n] # Equivalent to argmax returning the index of the n maxmimum values
        
        position_prediction_all = self.positionClassifier.predict(position_image)
        
        position_prediction = np.flip(np.argsort(position_prediction_all.flatten()))[0:n]

        return ([self.shape_vocabulary[x] for x in shape_prediction], [self.position_vocabulary[x] for x in position_prediction])
