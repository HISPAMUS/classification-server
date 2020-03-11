import logging
import threading
import datetime

from Classifier import *
from modelTemplates.simpleLanalysis import SimpleLayoutAnalysisScript
from datetime import date

import json
import os
from zipfile import ZipFile

__all__ = ['ModelManager']

class ModelManager:
    
    logger = logging.getLogger('ModelManager')

    e2eModels = dict()
    symbolclassificators = dict()
    documentAnalysismodels = dict()
    foldercorrespondence = dict()

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

        self.foldercorrespondence['eAgnosticSymbols'] = "model/symbol-classification"
        self.foldercorrespondence['eAgnosticEnd2End'] = "model/end-to-end"

    
    def getE2EModel(self, e2eModel, trained_with):

        e2eModelToReturn = None

        ######################################/
        folderPath = self.E2EPath + e2eModel
        ######################################/

        with self.mutex_lock:
         
            if e2eModel in self.e2eModels:
                self.logger.info('E2E Model exists in memory, returning it...')
                e2eModelToReturn = self.e2eModels[e2eModel]
                
            else:
                vocabularyToUse = self.vocabularyE2E
                self.logger.info('E2E Model does not exist in memory, loading it...')
                
                #######################################/
                #modelPath = folderPath + e2eModel + '.meta'
                #######################################/
                
                for file in os.listdir(folderPath):
                    if file.endswith(".npy") or file.endswith(".txt"):
                        vocabularyToUse = folderPath + file
                
                ###################################################/# 
                newE2EModel = End2EndClassifier(folderPath, trained_with)
                ###################################################/# 
                self.e2eModels[e2eModel] = newE2EModel
                e2eModelToReturn = newE2EModel
        
        return e2eModelToReturn
    
    def getDocumentAnalysisModel(self, documentAnalysisModel):
        documentAnalysisReturn = None
        
        with self.mutex_lock:
            
            if documentAnalysisModel in self.documentAnalysismodels:
                self.logger.info('Document Analysis Model exists in memory, returning it...')
                documentAnalysisReturn = self.documentAnalysismodels[documentAnalysisModel]
            
            else:
                ###################################################/
                newDocumentAnalysisModel = DocumentAnalysis()
                ###################################################/
                self.documentAnalysismodels[documentAnalysisModel] = newDocumentAnalysisModel
                documentAnalysisReturn = newDocumentAnalysisModel
        
        return documentAnalysisReturn

    
    def getSymbolClassifierModel(self, symbolClassName, trained_with):
        
        symbolModel = None

        with self.mutex_lock:

            if symbolClassName in self.symbolclassificators:
                self.logger.info('Symbol Classificator exists in memory, returning it...')
                symbolModel = self.symbolclassificators[symbolClassName]
            else:
                self.logger.info('Symbol Classificator does not exist in memory, loading it...')
                
                ###################################################################################/
                #modelPositionPath = self.SymbolPath + symbolClassName + "/" + symbolClassName + '_position.h5'
                #modelShapePath = self.SymbolPath + symbolClassName + "/" + symbolClassName + '_shape.h5'
                #vocabularyPosition = self.SymbolPath + symbolClassName + "/" + symbolClassName + '_position_map.npy'
                #vocabularyShape = self.SymbolPath + symbolClassName + "/" + symbolClassName + '_shape_map.npy'
                folder_Path = self.SymbolPath + symbolClassName
                ###################################################################################/

                #######################################################################################################/# 
                newSymbolClassModel = SymbolClassifier(folder_Path, trained_with)
                #######################################################################################################/# 

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
            
            #for key in list(self.symbolclassificators):
            #    elapsed_time = nowtime - self.symbolclassificators[key].getLastUsed()
            #    if (elapsed_time.seconds > self.eraseLimit):
            #        self.logger.info('Erasing unused model')
            #        del self.symbolclassificators[key]
        
        threading.Timer(60.0* self.waitTime, self.checkStatus).start()
    
    def getModelList(self, prefix, notationType, manuscriptType, collection, document, classifier):
        defpath = "db/" + prefix + notationType + "/" + manuscriptType + "/" 
        finalList = []
        #First I put the document's specific models
        #Need to check if the document requested exists or sth 
        self.searchByDocument(finalList, defpath, collection, document, classifier)
        
        self.searchInDirandAppendtoList(defpath, finalList, classifier)

        response = self.erase_duplicates(finalList)

        return response

    def erase_duplicates(self,listofdata):
        seen = set()
        return_list = []
        for data in listofdata:
            tup_data = tuple(data.items())
            if tup_data not in seen:
                seen.add(tup_data)
                return_list.append(data)

        return return_list        
    
    def searchByDocument(self, listToUse, defpath, collection, document, classifier):
        
        collectionPath = defpath + collection + "/"
        documentPath = defpath + collection + "/" + document + "/"
        self.logger.info(documentPath)
        if os.path.isdir(documentPath):
            self.searchInDirandAppendtoList(documentPath, listToUse, classifier)
        
        if os.path.isdir(collectionPath):
            self.searchInDirandAppendtoList(collectionPath, listToUse, classifier)


    def searchInDirandAppendtoList(self, dirToSearch,listToappend, classifier):
        for file in os.listdir(dirToSearch):
            if file.endswith(".json") and not file.startswith('.'):
                data = self.__loadJSON(dirToSearch+file)
                if not data == None:
                    if classifier == None or classifier == data['classifier_type']:
                        listToappend.append(data)
                
    
    def registerNewModel(self, name, classifier_type, notation_type, manuscript_type, collection, document, modelFile):
        
        modelid = ""
        vocabulary = ""
        with ZipFile(modelFile, 'r') as file:
            filenames = file.namelist()
            for names in filenames:
                if names.endswith(".index"):
                    modelid = names.split(".")[0]
                if names.endswith(".npy"):
                    vocabulary = names.split(".")[0]
                if names.endswith("h5"):
                    pre = names.split(".")[0]
                    modelid = names.split("_")[0]
        
        self.storeNewModel(modelid, classifier_type, modelFile)
        self.indexNewModel(modelid, name, classifier_type, notation_type, manuscript_type, collection, document, vocabulary)

        return
    
    def eraseModel(self, id):
        #Search for model and type. Supposing that a model is unique in our environment, we have to locate the first document who has it, we will deindex it later
        modelType = None
        for root, _, files in os.walk("data/"):
            for name in files:
                if name.endswith(".json"):
                    data = self.__loadJSON(os.path.join(root, name))
                    if data["id"] == id:
                        modelType = data["classifier_type"]
                        break
                
            if modelType is not None:
                break
        
        #We should have both the model type and name for the erasing



        return (modelType is None) #I return the condition because if it does not exist, we won't do anything
    
    def storeNewModel(self, modelid, classifier_type, modelfile):
        
        store_path = self.foldercorrespondence[classifier_type] + "/" + modelid

        try:
            os.mkdir(store_path)
        except FileExistsError:
            self.logger.info('Created folder already exists')

        with ZipFile(modelfile, 'r') as file:
            file.extractall(store_path)
        
        return


    def indexNewModel(self, modelid, name, classifier_type, notation_type, manuscript_type, collection, document, vocabulary):
        
        path_to_store = "db/" + notation_type + "/" + manuscript_type + "/"
        if not collection == None and not collection == "-1":
            path_to_store += collection + "/"
            try:
                os.mkdir(path_to_store)
            except FileExistsError:
                pass
            if not document == None and not document == "-1":
                path_to_store += document + "/"
                try: 
                    os.mkdir(path_to_store)
                except FileExistsError:
                    pass

        
        self.logger.info("Storing info at: " + path_to_store)

        data = {
            "name" : name,
            "id" : modelid,
            "last_train" : str(date.today()),
            "vocabulary" : vocabulary,
            "classifier_type": classifier_type
        }

        file = path_to_store + name + ".json"

        with open(file, 'w') as out:
            json.dump(data, out)
            

    def __loadJSON(self,pathToOpen):
        try:
            with open(pathToOpen) as model_data:
                data = json.load(model_data)
                return data
        except Exception:
            logging.info("File " + pathToOpen + " could not be opened")
            return None







