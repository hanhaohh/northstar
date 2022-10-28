import os
import time
from celery import Celery
from jobs.train import main
from celery_app import celery_client


@celery_client.task(name='tasks.add')
def add(x: int, y: int) -> int:
    time.sleep(10)
    return x + y


@celery_client.task(name='tasks.train')
def fit():
    task = celery_client.tasks["jobs.ml_model_task.IrisModel"]
    result = task.run()
    return result

