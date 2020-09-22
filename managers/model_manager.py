import json
import os
import threading
from .model_templates.e2e_model_tf import E2E_TF

from logger import Logger

####################### CONSTS ##############################################################

E2EPATH     = "models/end-to-end/"
SYMBOLPATH  = "models/symbol-classification/"

#############################################################################################
available_models = dict()
mutex_lock = threading.Lock()

logger_term = Logger()

def getE2EModel(model_id):
    global available_models
    folderpath = E2EPATH
    
    with mutex_lock:

        if model_id in available_models:
            logger_term.LogInfo(f"The requested model {model_id} exists in memory, returning it")
            return available_models[model_id]
        else:
            logger_term.LogInfo(f"The requested model {model_id} does not exist in memory, retrieving it")
            vocabulary = ""
            
            modelpath  = folderpath + model_id + "/"
            
            logger_term.LogInfo(f"The model path is {modelpath}")

            for file in os.listdir(modelpath):
                if file.endswith(".npy") or file.endswith(".txt"):
                    vocabulary = modelpath + file

            e2eModel = E2E_TF(model_path=modelpath + model_id + ".meta", w2i=vocabulary)
            available_models[model_id] = e2eModel
            logger_term.LogInfo("Model loaded correctly")
            return e2eModel

    return None

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
