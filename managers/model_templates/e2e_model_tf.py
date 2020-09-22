from .model import Model
import tensorflow as tf
import numpy as np
import cv2

import logging
from logger import Logger

logger_term = Logger()

class E2E_TF(Model):

    def __init__(self, *args, **kwargs):
        try:     
            model_path = kwargs["model_path"]
            dictionary = np.load(kwargs["w2i"], allow_pickle=True).item()
            self.i2w = dict((v,k) for k, v in dictionary.items())
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.reset_default_graph()

            self.session = tf.InteractiveSession(config = config)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(self.session, model_path[:-5])
    
            graph = tf.get_default_graph()
            self.input = graph.get_tensor_by_name('model_input:0')
            self.seq_len = graph.get_tensor_by_name("seq_lengths:0")
            self.rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
            height_tensor = graph.get_tensor_by_name("input_height:0")
            width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
            logits = tf.get_collection("logits")[0]

            # Constants that are saved inside the model itself
            self.WIDTH_REDUCTION, self.HEIGHT = self.session.run([width_reduction_tensor, height_tensor])

            #decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
            self.decoded = tf.nn.softmax(logits)
        except Exception as e:
            logger_term.LogInfo(e)


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

            seq_lengths = [ image.shape[2] / self.WIDTH_REDUCTION ]

            prediction = self.session.run(self.decoded,
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
                if w < len(self.i2w): # != BLANK
                    if prev == -1:
                        start = idx
                    elif prev != w:
                        if prev != len(self.i2w):
                            result.append((self.i2w[prev],int(start*width_refactor),int(idx*width_refactor)))
                        start = idx
                else: # == BLANK
                    if prev != -1 and prev != len(self.i2w):
                        result.append((self.i2w[prev],int(start*width_refactor),int(idx*width_refactor)))
                prev = w

            # for symbol, start, end in result:
            #     self.logger.info(f'{symbol} {start} {end}')
            return result
        except Exception as e:
            logger_term.LogError(e)
	
	
