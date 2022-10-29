import os
import json
import time
from celery_app import celery_client
from sqlalchemy import create_engine
from jobs.data_modified.file_metadata import FileMetadata
from sqlalchemy.orm import sessionmaker


engine = create_engine(
    "mysql+pymysql://root:123@mysqldb:3306/file_activity",
    echo=True,
)


@celery_client.task(name='tasks.add')
def add(x: int, y: int) -> int:
    time.sleep(2)
    return x + y


@celery_client.task(name='tasks.train')
def fit():
    task = celery_client.tasks["jobs.ml_model_task.IrisModel"]
    result = task.run()
    return result


@celery_client.task(name='tasks.data_change')
def data_change(event_type, src_path, md5):
    file_name = os.path.basename(src_path)
    Session = sessionmaker(bind=engine)
    session = Session()
    instance = session.query(FileMetadata).filter_by(file_md5=md5).first()
    if instance:
        return instance.file_md5
    else:
        instance = FileMetadata(file_name=file_name, file_path=src_path, file_md5=md5)
        session.add(instance)
        session.commit()
        fit.delay()
        return instance.file_md5
