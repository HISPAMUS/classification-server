import logging

class Logger:
    def __init__(self):
        self.infoLogger = logging.getLogger("uvicorn.info")
        self.errorLogger = logging.getLogger("uvicorn.error")
        self.warningLogger = logging.getLogger("uvicorn.warning")

    def LogInfo(self, msg):
        self.infoLogger.info(msg)
    
    def LogError(self, msg):
        self.errorLogger.error(msg)
    
    def LogWarning(self, msg):
        self.errorLogger.warning(msg)
        
