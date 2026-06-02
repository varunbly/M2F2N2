from abc import ABC, abstractmethod

class BaseMetaMethod(ABC):
    @abstractmethod
    def train_step(self, model, support, query, inner_lr, inner_steps, lam):
        """Execute one meta-training step on a task and return query loss."""
        pass

    @abstractmethod
    def adapt(self, model, support, inner_lr, inner_steps, lam):
        """Returns an 'adapted state' (e.g. cloned & adapted model)."""
        pass

    @abstractmethod
    def predict(self, adapted_state, price, news, tf):
        """Generates positions given adapted state."""
        pass
