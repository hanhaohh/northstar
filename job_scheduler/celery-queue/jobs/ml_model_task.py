"""Celery task for making ML Model predictions."""
import importlib
from celery import Task


class MLModelTask(Task):
    """Celery Task for making ML Model predictions."""

    def __init__(self, module_name, class_name):
        """Class constructor."""
        super().__init__()
        self._model = None

        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)

        # if issubclass(model_class, MLModel) is False:
        #     raise ValueError("MLModelTask can only be used with subtypes of MLModel.")

        # saving a reference to the class to avoid having to import it again
        self._model_class = model_class

        # dynamically adding a name to the task object
        self.name = "{}.{}".format(__name__, model_class.__name__)
        
    def initialize(self):
        """Class initialization."""
        model_object = self._model_class()
        self._model = model_object

    def run(self):
        """Execute predictions with the MLModel class."""
        if self._model is None:
            self.initialize()
        return self._model.fit()