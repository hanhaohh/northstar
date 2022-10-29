import os
from watchdog.events import PatternMatchingEventHandler
import hashlib
from celery_client import celery_client


class MyHandler(PatternMatchingEventHandler):
    
    FILE_SIZE = 1000000000

    def __init__(self):
        PatternMatchingEventHandler.__init__(
            self,
            patterns=['*.csv', ".tsv"],
            ignore_directories=True, case_sensitive=False
        )

    def on_modified(self, event):
        path = event.src_path
        file_size = os.path.getsize(path)

        if file_size > self.FILE_SIZE:
            print("Time to backup the dir")
        md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
        print(f'event type: {event.event_type}  path : {event.src_path} md5 : {md5}')
        task = celery_client.send_task('tasks.data_change', args=[event.event_type, event.src_path, md5], kwargs={})
        
    def on_created(self, event):
        self.checkFolderSize(event.src_path)
    
    def on_deleted(self, event):
        print(event.src_path, event.event_type)
               
    def checkFolderSize(self, src_path):
        if os.path.isdir(src_path):
            if os.path.getsize(src_path) > self.FILE_SIZE:
                print("Time to backup the dir")
        else:
            if os.path.getsize(src_path) > self.FILE_SIZE:
                print("very big file, needs to be backed up")
