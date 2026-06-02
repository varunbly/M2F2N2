from abc import ABC, abstractmethod

class BaseDataSource(ABC):
    @abstractmethod
    def fetch_all(self):
        """Fetch all required data."""
        pass
