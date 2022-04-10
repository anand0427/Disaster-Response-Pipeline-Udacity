# Disaster Response Pipeline Project

### Idea and Design
The purpose of this project is to create and serve a classifier as a service for disaster response. An input message is given to the service which will perform a few necessary cleanup of the input and pass the data to a classifier which will return a set of classes that the message belongs to. 

The model is trained from in-house data which is available in the data/ directory. The data which is available as a csv file is processed and pushed into a sqldb. The sqldb is then called into the workspace to train a MultiOutputClassifier. 

The classifier is saved as a pickle model in the models/ directory and called into a Flask app to serve in an API. 

The Flask app also serves a plotly Bar graph to visualize details from dataframe like related, aid etc.

### Dependencies

1. Software dependencies
    - Python
    - Python Environment
2. Library dependencies
    - Flask
    - sklearn
    - nltk
    - plotly
    - json
    - pandas
    - sqlalchemy

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
