# NorthStar

Once your machine learning model is stable and but data is not, so you would need to constantly tune your model based any change of the data. NorthStar is a data centric machine learning platform where it tracks any change to your data, kick off training and evaluation automatically.

It has three main components:
1. Job server system using Celery task queue - defining how we train and eval model.
2. Data programming system. It tracks data change and also allow you to program data from GUI or through API.
3. Metrics reporting. Report model performance based on different version of data.



