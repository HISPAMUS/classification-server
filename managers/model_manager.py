import json
import os
import threading
from .model_templates.e2e_model_tf import E2E_TF
from .model_templates.doc_analysis_model_k import Document_Analysis_K
from .model_templates.simple_document_analysis import SimpleDocumentAnalysisScript
from .model_templates.symbols_model_k import SymbolsModel
from .routines.document_analysis_routines import predict_regions


from logger import Logger

import enum

####################### CONSTS ##############################################################
MODELPATHS = {
    "E2E" : "models/end-to-end/",
    "DOCANALYSIS" : "models/document-analysis/",
    "SYMBOLS"  : "models/symbol-classification/"
}

#############################################################################################
available_models = dict()
mutex_lock = threading.Lock()

logger_term = Logger()


def getE2EModel(model_id):
    e2eModel = None
    folderpath = MODELPATHS["E2E"]
    
    logger_term.LogInfo(f"Loading {model_id}")
    vocabulary = ""        
    modelpath  = folderpath + model_id + "/"
    logger_term.LogInfo(f"The model path is {modelpath}")

    for file in os.listdir(modelpath):
        if file.endswith(".npy") or file.endswith(".txt"):
            vocabulary = modelpath + file

    e2eModel = E2E_TF(model_path=modelpath + model_id + ".meta", w2i=vocabulary)
    logger_term.LogInfo("Model loaded correctly")
    
    return e2eModel


def getDocumentAnalysisModel(model_id):
    folderpath = MODELPATHS["DOCANALYSIS"]
        
    logger_term.LogInfo(f"Loading {model_id}")

    docAnalysisModel = None
            
    if model_id == "simple-lan": docAnalysisModel = SimpleDocumentAnalysisScript()
    else: docAnalysisModel = Document_Analysis_K(folderpath + model_id + "/" + model_id + ".h5")
        
    return docAnalysisModel


def getSymbolsRecogintionModel(model_id):
    symbolModel = None
    
    modelPositionPath = MODELPATHS["SYMBOLS"] + model_id + "/" + model_id + '_position.h5'
    modelShapePath = MODELPATHS["SYMBOLS"] + model_id + "/" +  model_id + '_shape.h5'
    vocabularyPosition = MODELPATHS["SYMBOLS"] + model_id + "/" + model_id + '_position_map.npy'
    vocabularyShape = MODELPATHS["SYMBOLS"] + model_id + "/" + model_id + '_shape_map.npy'
    
    symbolModel = SymbolsModel(modelShapePath, modelPositionPath, vocabularyShape, vocabularyPosition)
        
    return symbolModel



###############################################################################################
# MODEL LISTING METHODS
###############################################################################################
def erase_duplicates(listofdata):
        seen = set()
        return_list = []
        for data in listofdata:
            tup_data = tuple(data.items())
            if tup_data not in seen:
                seen.add(tup_data)
                return_list.append(data)

        return return_list        

def loadJSON(pathToOpen):
        try:
            with open(pathToOpen) as model_data:
                data = json.load(model_data)
                return data
        except Exception:
            logging.info("File " + pathToOpen + " could not be opened")
            return None

def searchInDirandAppendtoList(dirToSearch,listToappend, classifier):
    for file in os.listdir(dirToSearch):
        if file.endswith(".json") and not file.startswith('.'):
            data = loadJSON(dirToSearch+file)
            if not data == None:
                if classifier == None or classifier == data['classifier_type']:
                    listToappend.append(data)


def searchByDocument(listToUse, defpath, collection, document, classifier):
    collectionPath = defpath + collection + "/"
    documentPath = defpath + collection + "/" + document + "/"
    
    if os.path.isdir(documentPath):
        searchInDirandAppendtoList(documentPath, listToUse, classifier)
        
    if os.path.isdir(collectionPath):
        searchInDirandAppendtoList(collectionPath, listToUse, classifier)


def getModelList(prefix, notationType, manuscriptType, collection, document, classifier):
    defpath = "db/" + prefix + notationType + "/" + manuscriptType + "/" 
    finalList = []
    #First I put the document's specific models
    #Need to check if the document requested exists or sth 
    searchByDocument(finalList, defpath, collection, document, classifier)
        
    searchInDirandAppendtoList(defpath, finalList, classifier)

    response = erase_duplicates(finalList)

    return response
##################################################################################################
