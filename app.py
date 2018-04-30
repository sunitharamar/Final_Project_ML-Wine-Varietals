from flask import Flask, jsonify, g, render_template, request
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import sqlalchemy
import numpy as np
import pandas as pd
import json
import pickle

dialect = 'sqlite'
port = 3306
database = 'wines.sqlite'

engine = create_engine(f'{dialect}:///{database}')
# reflect an existing database into a new model

Base = automap_base()
Base.prepare(engine, reflect=True)
print(Base.classes.keys())

session = Session(bind=engine)

Wines = Base.classes.wines

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer # term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

filename = "Logistic_Model.sav"
loaded_model = pickle.load(open(filename, 'rb'))

stop_word_df = pd.read_sql('stopWords', con=engine)
stop_word_list = list(stop_word_df['stopWords'])
vect = TfidfVectorizer(stop_words=stop_word_list)

wine_data = pd.read_sql('wines', con=engine)
X = vect.fit_transform(wine_data['description'])

le = LabelEncoder()
y = le.fit_transform(wine_data['variety'])

predResult = ""
searchTerms = ""

#################################################
# Flask Setup
#################################################

app = Flask(__name__)

#################################################
# Flask Routes
#################################################

@app.route("/", methods=["GET", "POST"])
def home():
    print(loaded_model)
    predResult = ""
    searchTerms = ""
    probs = ""
    
    if request.method == "POST":
        print(request.form)
        searchTerms = request.form['searchTerms']
        model_pred = vect.transform([searchTerms])

        probab = loaded_model.predict_proba(model_pred)
        best_n = np.argsort(probab, axis=1)[:,-10:]
        probs = sorted(probab[0], reverse=True)[0:10]
        print(sorted(probab[0], reverse=True)[0:10])
        print([searchTerms])
        predResult = [le.classes_[best_n[0][9]],le.classes_[best_n[0][8]],le.classes_[best_n[0][7]],le.classes_[best_n[0][6]],le.classes_[best_n[0][5]],
        le.classes_[best_n[0][4]],le.classes_[best_n[0][3]], le.classes_[best_n[0][2]],le.classes_[best_n[0][1]],le.classes_[best_n[0][0]]]
        print(le.classes_[best_n[0][4]])
        print(le.classes_[best_n[0][3]])
        print(le.classes_[best_n[0][2]])
        print(le.classes_[best_n[0][1]])
        print(le.classes_[best_n[0][0]])

    return render_template("index.html", predResult = predResult, searchTerms = searchTerms, probs=[probs])

@app.route("/data/")
def data():
    wine_data = pd.read_sql('wines', con=engine).to_dict("records")
    return jsonify(wine_data)
@app.route("/stopWords/")
def stopWords():
    stop_words = pd.read_sql('stopWords', con=engine).to_dict("records")
    return jsonify(stop_words)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)