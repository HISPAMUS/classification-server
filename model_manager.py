import logging
import threading
import datetime

from e2e_classifier import E2EClassifier

__all__ = ['ModelManager']

class ModelManager:
    
    logger = logging.getLogger('ModelManager')

    e2eModels = dict()
    symbolclassificators = dict()

    vocabularyE2E = ''

    E2EPath = 'model/end-to-end/'

    mutex_lock = threading.Lock() #Just secure that two or several threads don't mess up our dictionaries, they are shared resources who could be corrupted

    def __init__(self, defaultVocabularyE2E):
        
        self.logger.info('Model Manager Initialized')
        self.vocabularyE2E = defaultVocabularyE2E
        threading.Timer(60.0, self.checkStatus).start()
    
    def getE2EModel(self, e2eModel):

        e2eModelToReturn = None

        with self.mutex_lock:
         
            if e2eModel in self.e2eModels:
                self.logger.info('E2E Model exists in memory, returning it...')
                e2eModelToReturn = self.e2eModels[e2eModel]
                
            else:
                self.logger.info('E2E Model does not exist in memory, loading it...')
                modelPath = self.E2EPath + e2eModel + '.meta'
                newE2EModel = E2EClassifier(modelPath, self.vocabularyE2E)
                self.e2eModels[e2eModel] = newE2EModel
                e2eModelToReturn = newE2EModel
        
        return e2eModelToReturn
    
    def checkStatus(self):

        with self.mutex_lock:
        
            self.logger.info('Checking for unused models:')

            for key in list(self.e2eModels):
                nowtime = datetime.datetime.now()
                elapsed_time = nowtime - self.e2eModels[key].getLastUsed()
                if (elapsed_time.seconds > 60.0):
                    self.logger.info('Erasing the unused model')
                    #del self.e2eModels[key].value
                    del self.e2eModels[key]
        
        
        threading.Timer(60.0, self.checkStatus).start()





