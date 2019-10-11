import logging
import threading
import datetime

from e2e_classifier import E2EClassifier
from symbol_classifier import SymbolClassifier

import json
import os

__all__ = ['ModelManager']

class ModelManager:
    
    logger = logging.getLogger('ModelManager')

    e2eModels = dict()
    symbolclassificators = dict()

    vocabularyE2E = ''
    
    vocabularyShape = ''
    vocabularyPos= ''

    E2EPath = 'model/end-to-end/'
    SymbolPath = 'model/symbol-classification/'

    mutex_lock = threading.Lock() #Just secure that two or several threads don't mess up our dictionaries, they are shared resources who could be corrupted

    waitTime = 60.0
    eraseLimit = 30.0 * 60.0

    def __init__(self, defaultVocabularyE2E, defaultVocabularyShape, defaultVocabularyPos):
        
        self.logger.info('Model Manager Initialized')
        self.vocabularyE2E = defaultVocabularyE2E
        self.vocabularyShape = defaultVocabularyShape
        self.vocabularyPos = defaultVocabularyPos
        threading.Timer(60.0 * self.waitTime, self.checkStatus).start()
    
    def getE2EModel(self, e2eModel):

        e2eModelToReturn = None
        folderPath = self.E2EPath + e2eModel + "/"

        with self.mutex_lock:
         
            if e2eModel in self.e2eModels:
                self.logger.info('E2E Model exists in memory, returning it...')
                e2eModelToReturn = self.e2eModels[e2eModel]
                
            else:
                vocabularyToUse = self.vocabularyE2E
                self.logger.info('E2E Model does not exist in memory, loading it...')
                modelPath = folderPath + e2eModel + '.meta'
                
                for file in os.listdir(folderPath):
                    if file.endswith(".npy") or file.endswith(".txt"):
                        vocabularyToUse = folderPath + file
                
                newE2EModel = E2EClassifier(modelPath, vocabularyToUse)
                self.e2eModels[e2eModel] = newE2EModel
                e2eModelToReturn = newE2EModel
        
        return e2eModelToReturn
    
    def getSymbolClassifierModel(self, symbolClassName):
        
        symbolModel = None

        with self.mutex_lock:

            if symbolClassName in self.symbolclassificators:
                self.logger.info('Symbol Classificator exists in memory, returning it...')
                symbolModel = self.symbolclassificators[symbolClassName]
            else:
                self.logger.info('Symbol Classificator does not exist in memory, loading it...')
                modelPositionPath = self.SymbolPath + symbolClassName + '_position.h5'
                modelShapePath = self.SymbolPath + symbolClassName + '_shape.h5'
                newSymbolClassModel = SymbolClassifier(modelShapePath, modelPositionPath, self.vocabularyShape, self.vocabularyPos)
                self.symbolclassificators[symbolClassName] = newSymbolClassModel
                symbolModel = newSymbolClassModel
            
        return symbolModel

    
    def checkStatus(self):

        nowtime = datetime.datetime.now()

        with self.mutex_lock:
        
            self.logger.info('Checking for unused models:')

            for key in list(self.e2eModels):
                elapsed_time = nowtime - self.e2eModels[key].getLastUsed()
                if (elapsed_time.seconds > 30.0):
                    self.logger.info('Erasing the unused model')
                    #del self.e2eModels[key].value
                    del self.e2eModels[key]
            
            for key in list(self.symbolclassificators):
                elapsed_time = nowtime - self.symbolclassificators[key].getLastUsed()
                if (elapsed_time.seconds > self.eraseLimit):
                    self.logger.info('Erasing unused model')
                    del self.symbolclassificators[key]
        
        threading.Timer(60.0* self.waitTime, self.checkStatus).start()
    
    def getModelList(self, notationType, manuscriptType, collection, project, classifier):
        defpath = "db/" + notationType + "/" + manuscriptType + "/" 
        finalList = []
        #First I put the project's specific models
        #Need to check if the project requested exists or sth 
        self.searchByProject(finalList, defpath, collection, project, classifier)
        
        self.searchInDirandAppendtoList(defpath, finalList, classifier)
        
        return finalList
    
    def searchByProject(self, listToUse, defpath, collection, project, classifier):
        
        projectPath = ""
        if collection != None and project != None:
            self.logger.info('Searching by project and collection')
            projectPath = defpath + collection + "/" + project + "/"
            self.logger.info(projectPath)
            if os.path.isdir(projectPath):
                self.searchInDirandAppendtoList(projectPath, listToUse, classifier)
        elif collection != None:
            self.logger.info('Searching by collection only')
            projectPath = defpath + collection + "/"
            for _, dirs, _ in os.walk(projectPath):
                for directory in dirs:
                    self.searchInDirandAppendtoList(projectPath+directory+"/", listToUse, classifier)

    
    def searchInDirandAppendtoList(self, dirToSearch,listToappend, classifier):
        for file in os.listdir(dirToSearch):
            if file.endswith(".json"):
                data = self.__loadJSON(dirToSearch+file)
                if classifier == None or classifier == data['classifier_type']:
                    listToappend.append(data)

    def __loadJSON(self,pathToOpen):
        with open(pathToOpen) as model_data:
            data = json.load(model_data)
            return data







