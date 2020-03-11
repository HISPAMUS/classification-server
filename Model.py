import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


class Model:

    def __init__(self, model_path, vocabulary_path, trained):

        self.trained = trained

        if self.trained == "Keras":
            self._init_keras(model_path, vocabulary_path)
        if self.trained == "Tensor Flow":
            self._init_TF(model_path, vocabulary_path)


    def _init_keras(self, model_path, vocabulary_path):
        self.model = load_model(model_path)
        self.model._make_predict_function()

        vocabulary = np.load(vocabulary_path, allow_pickle=True).item()  # Category -> int
        self.word2int = vocabulary
        self.vocabulary = dict((v, k) for k, v in vocabulary.items())  # int -> Category
        self.int2word = self.vocabulary

        self.vocabularyLength = len(vocabulary)

        self.HEIGHT = 128

    
    def _init_TF(self, model_path, vocabulary_path):
        self.model_path = model_path
        # Read the dictionary
        word2int = np.load(vocabulary_path, allow_pickle=True).item()     # Category -> int
        self.int2word = dict((v, k) for k, v in word2int.items())     # int -> Category

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession(config=config)

        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(self.model_path)
        saver.restore(self.sess, self.model_path[:-5])

        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name('model_input:0')
        self.seq_len = graph.get_tensor_by_name("seq_lengths:0")
        self.rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = tf.get_collection("logits")[0]

        # Constants that are saved inside the model itself
        self.WIDTH_REDUCTION, self.HEIGHT = self.sess.run([width_reduction_tensor, height_tensor])

        #decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
        self.decoded = tf.nn.softmax(logits)


    def predict(self, img):
        if self.trained == "Keras":
            return self._predict_keras(img)
        if self.trained == "Tensor Flow":
            return self._predict_TF(img)
        
        return None


    def e2e_predict(self, img):
        if self.trained == "Keras":
            return self._predict_keras_e2e(img)
        if self.trained == "Tensor Flow":
            return self._predict_TF(img)

        return None


    def symbol_predict(self, img):
        if self.trained == "Keras":
            return self.predict(img)
        if self.trained == "Tensor Flow":
            return self._predict_TF_symbol(img)

        return None


    def _predict_keras(self, img):
        # Predictions

        n = 1

        model_prediction_all = self.model.predict(img)

        # Equivalent to argmax returning the index of the n maxmimum values
        model_prediction = np.flip(np.argsort(model_prediction_all.flatten()))[0:n]
        model_predicted = [self.vocabulary[x] for x in model_prediction]

        return model_predicted

    def _predict_keras_e2e(self, img):
        original_image_shape = img.shape
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Pre-process
        new_width = int(float(self.HEIGHT * img.shape[1]) / img.shape[0])
        img = cv2.resize(img, (new_width, self.HEIGHT))

        

        img = (255.-img)/255
        img = np.asarray(img).reshape(1,img.shape[0],img.shape[1],1)

        result = self._test_prediction(img, self.model, self.int2word)

        return result


    def _predict_TF(self, img):
         #original_image = cv2.imread(image_path,True)
        original_image_shape = img.shape
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Pre-process
        new_width = int(float(self.HEIGHT * img.shape[1]) / img.shape[0])
        img = cv2.resize(img, (new_width, self.HEIGHT))

        img = (255.-img)/255
        img = np.asarray(img).reshape(1,img.shape[0],img.shape[1],1)

        seq_lengths = [ img.shape[2] / self.WIDTH_REDUCTION ]

        prediction = self.sess.run(self.decoded,
                            feed_dict={
                                self.input: img,
                                self.seq_len: seq_lengths,
                                self.rnn_keep_prob: 1.0,
                            })

        # prediction -> [frame, sample, character]

        pred_per_frame = []
        for i, v in enumerate(prediction):
                pred_per_frame.append(np.argmax(v[0]))

        width_refactor = 1.
        width_refactor *= original_image_shape[0]
        width_refactor /= self.HEIGHT
        width_refactor *= self.WIDTH_REDUCTION
        
        # [!] Force the frames to span over the whole width of the image
        # width_refactor = (1.*image.shape[2])/len(pred_per_frame)

        # Process symbol and positions
        result = []
        prev = -1
        for idx, w in enumerate(pred_per_frame):
            if w < len(self.int2word): # != BLANK
                if prev == -1:
                    start = idx
                elif prev != w:
                    if prev != len(self.int2word):
                        result.append((self.int2word[prev],int(start*width_refactor),int(idx*width_refactor)))
                    start = idx
            else: # == BLANK
                if prev != -1 and prev != len(self.int2word):
                    result.append((self.int2word[prev],int(start*width_refactor),int(idx*width_refactor)))
            prev = w

        # for symbol, start, end in result:
        #     self.logger.info(f'{symbol} {start} {end}')
        return result

    def _predict_TF_symbol(self, img):
        return None




    def _test_prediction(self, sequence, model, i2w):
        decoded = np.zeros((1,500,self.vocabularyLength), dtype=np.float)
        decoded_input = np.asarray(decoded)
        prediction = model.predict([sequence, decoded_input])
        predicted_sequence = [i2w[char] for char in np.argmax(prediction[0], axis=1)]
        predicted = []
        
        for char in predicted_sequence:
            predicted += [char]
            if char == '</s>':
                break

        return predicted


    def __del__(self):
        if self.trained == "Tensor Flow":
            self.sess.close()