from abc import ABC, abstractmethod


class PreTransform(ABC):
    """
    Transform for pretraining task.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, image, mask, dist):
        raise NotImplementedError
