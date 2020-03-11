import cv2
import logging
import os
import datetime
import sys
from skimage.filters import threshold_sauvola
from image_storage import ImageStorage
from Model import Model


__all__ = ['Classifier', 'End2EndClassifier', 'SymbolClassifier', 'DocumentAnalysis', 'Encoder']

class Classifier:
    
    lastUsed = None
    modelFormat = None
    modelShape = None
    modelPosition = None
    vocabularyFormat = "npy"
    modelsNumber = 1


    def __init__(self, folder_model, trained):
        self.logger.info('Loading models...')

        self.trained = trained

        if trained == "Keras":
            self.modelFormat = "h5"
        if trained == "Tensor Flow":
            self.modelFormat = "meta"

        files = os.listdir(folder_model)

        self.modelFiles = list()
        self.vocabularyFiles = list()

        for file in files:
            fileType = file.split(".")[-1]
            if fileType == self.modelFormat:
                self.modelFiles.append(folder_model + "/" + file)
            if fileType == self.vocabularyFormat:
                self.vocabularyFiles.append(folder_model + "/" + file)

    def getLastUsed(self):
        return self.lastUsed

    def __del__(self):

        if self.modelShape!=None:
            self.modelShape.__del__()
        
        if self.modelPosition!=None:
            self.modelPosition.__del__()

        self.logger.info('Object destroyed!')


class End2EndClassifier(Classifier):

    logger = logging.getLogger('End2EndClassifier')

    def __init__(self, folder_model, trained):
        super().__init__(folder_model, trained)

        model_path = self.modelFiles[0]
        vocabulary_path = self.vocabularyFiles[0]

        self.modelShape = Model(model_path, vocabulary_path, trained)
        self.modelPosition = None

        self.logger.info('Models loaded')
        self.lastUsed = datetime.datetime.now()
    
    def predict(self, img):
        print("A staff")
        if self.trained == "Keras":
            return self._keras_predict(img)
        if self.trained == "Tensor Flow":
            return self._TF_predict(img)

        return None

    def _keras_predict(self, img):
        return self.modelShape.e2e_predict(img)
    
    def _TF_predict(self, img):
        return self.modelShape.e2e_predict(img)


class SymbolClassifier(Classifier):

    logger = logging.getLogger('SymbolClassifier')

    def __init__(self, folder_model, trained):
        super().__init__(folder_model, trained)

        self.modelsNumber = 2

        model_path = ""
        vocabulary_path = ""
        model_p_path = ""
        vocabulary_p_path = ""

        for m in self.modelFiles:
            l = m.split("." + self.modelFormat)[-2].split("_")
            if "shape" in l:
                model_path = m
            if "position" in l:
                model_p_path = m
        
        for v in self.vocabularyFiles:
            l = v.split("." + self.vocabularyFormat)[-2].split("_")
            if "shape" in l:
                vocabulary_path = v
            if "position" in l:
                vocabulary_p_path = v

        self.modelShape = Model(model_path, vocabulary_path, self.trained)
        self.modelPosition = Model(model_p_path, vocabulary_p_path, self.trained)

        self.logger.info('Models loaded')
        self.lastUsed = datetime.datetime.now()
    
    def predict(self, imgS, imgP):
        if self.trained == "Keras":
            return self._keras_predict(imgS, imgP)
        if self.trained == "Tensor Flow":
            return self._TF_predict(imgS, imgP)

        return None

    def _keras_predict(self, imgS, imgP):
        return (self.modelShape.symbol_predict(imgS), self.modelPosition.symbol_predict(imgP))
    
    def _TF_predict(self, imgS, imgP):
        return (self.modelShape.symbol_predict(imgS), self.modelPosition.symbol_predict(imgP))


class DocumentAnalysis(Classifier):

    logger = logging.getLogger('DocumentAnalysis')

    def __init__(self):
        pass

    def predict(self, img):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = threshold_sauvola(img_gray, window_size=25)

        binary_img = np.asarray((img_gray < thresh)*255.).astype('uint8')

        erode_img = cv2.erode(binary_img,np.ones((1,20),np.uint8),iterations = 1)

        staff_img = cv2.dilate(erode_img,np.ones((1,20),np.uint8),iterations = 1)
        staff_img = cv2.dilate(staff_img,np.ones((50,1),np.uint8),iterations = 1)

        contours, _ = cv2.findContours(staff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_draw = img.copy()

        boundings = []

        for contour in contours:
            if self.isStaff(contour, img.shape):
                x,y,w,h = cv2.boundingRect(contour)
                boundings.append({"x0": x, "y0": y, "xf": (x+w), "yf": (y+h), "regionType": "staff"})
                print('Staff',x,y,w,h)
        
        return boundings

    def isStaff(self,contour, img_shape):
        x,y,w,h = cv2.boundingRect(contour)

        cond1 = h < img_shape[0] // 5
        cond2 = cv2.contourArea(contour) > 8000

        return cond1 and cond2


class Encoder(Classifier):

    logger = logging.getLogger('Encoder')

    def __init__(self):
        pass

    def predict(self):
        pass


#SymbolClassifier

modShape = "X_shape.(h5/meta)"
vocShape = "X_shape_map.npy"
modPosit = "X_position.(h5/meta)"
vocPosit = "X_position_map.npy"


keras_format = "h5"
tf_format = "meta"