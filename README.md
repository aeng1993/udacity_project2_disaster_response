# udacity_project2_disaster_response
Udacity project 2 - disaster response pipeline


## Installation
Note that a package called CountLength is included under the "models" folder. This is used to create a feature in the machine learning model pipeline. Pip install this package so that the code will work.

Several libraries are used in the code file:
sys
numpy
pandas 
sqlalchemy
nltk
sklearn
pickle
json
plotly
flask 
joblib


## Instruction
To set up the ETL pipeline and create the database:
run python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To set up the machine learning pipeline:
go to models folder, pip install the CountLength package
run python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

To run the web app:
run python run.py
go to http://0.0.0.0:3001/


## Overview
In this project, I built a disaster response pipeline to support the emergency response officials in classifying received messages. A dataset of messages and their corresponding categories is provided by Figure Eight. This pipeline first cleans the data and stores it in a database, then load the data to train and evalaute a machine learning model (random forest classifier), and finally use the model in a web app visualization. A user can type in a message in the web page and the classification of the message will be returned.

## Conclusion/Summary
The RandomForestClassifier (multi-output) has a weighted F1 score of 0.67 for all 36 categories in test set data.
The model has a F1 score close to 0.90 for classifying whether a message is related/relevant, but has low score for some of the more specific categories which have limited training data. The model performance is bad for those categories with less than 300 data point (before train/test split). This is likely due to the dataset is very imbalanced for those categories (negative takes majority >98%). No positive data for "child alone" category is available so the model cannot predict on this one.

