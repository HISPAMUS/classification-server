import cv2
import logging
import numpy as np
import os
import tensorflow as tf
import datetime

__all__ = [ 'E2EClassifier' ]

class E2EClassifier:

    logger = logging.getLogger('E2EClassifier')
    lastUsed = datetime.datetime.now()


    def __init__(self, model_path, vocabulary_path):
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


    # def sparse_tensor_to_strs(self, sparse_tensor):
    #     indices= sparse_tensor[0][0]
    #     values = sparse_tensor[0][1]
    #     dense_shape = sparse_tensor[0][2]

    #     strs = [ [] for i in range(dense_shape[0]) ]

    #     string = []
    #     ptr = 0
    #     b = 0

    #     for idx in range(len(indices)):
    #         if indices[idx][0] != b:
    #             strs[b] = string
    #             string = []
    #             b = indices[idx][0]

    #         string.append(values[ptr])

    #         ptr = ptr + 1

    #     strs[b] = string

    #     return strs


    def predict(self, image):

        self.lastUsed = datetime.datetime.now()

        #original_image = cv2.imread(image_path,True)
        original_image_shape = image.shape
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Pre-process
        new_width = int(float(self.HEIGHT * image.shape[1]) / image.shape[0])
        image = cv2.resize(image, (new_width, self.HEIGHT))

        image = (255.-image)/255
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

        seq_lengths = [ image.shape[2] / self.WIDTH_REDUCTION ]

        prediction = self.sess.run(self.decoded,
                            feed_dict={
                                self.input: image,
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

    
    def __del__(self):
        self.sess.close()
        self.logger.info('Object destroyed!')
    
    def getLastUsed(self):
        return self.lastUsed
