from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def getStateCount(self):
        pass
