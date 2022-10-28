import os
import jobs
from celery import Celery
from celery.app.registry import TaskRegistry

from jobs import __name__
from jobs.ml_model_task import MLModelTask
from jobs.config import Config


registry = TaskRegistry() 
celery_client = Celery(__name__, tasks=registry)


for model in Config.models:
    registry.register(
        MLModelTask(
            module_name=model["module_name"],
            class_name=model["class_name"]
        )
    )
