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
    log_predResult = ""
    searchTerms = ""
    log_probs = ""
    entered = "You entered: "

    if request.method == "POST":
        print(request.form)
        searchTerms = request.form['searchTerms']
        log_model_pred = vect.transform([searchTerms])

        log_probab = loaded_model.predict_proba(log_model_pred)
        log_best_n = np.argsort(log_probab, axis=1)[:,-10:]
        log_probs = sorted(log_probab[0], reverse=True)[0:10]
        print(sorted(log_probab[0], reverse=True)[0:10])
        print([searchTerms])
        log_predResult = [le.classes_[log_best_n[0][9]],le.classes_[log_best_n[0][8]],le.classes_[log_best_n[0][7]],le.classes_[log_best_n[0][6]],le.classes_[log_best_n[0][5]],
        le.classes_[log_best_n[0][4]],le.classes_[log_best_n[0][3]], le.classes_[log_best_n[0][2]],le.classes_[log_best_n[0][1]],le.classes_[log_best_n[0][0]]]
        print(le.classes_[log_best_n[0][4]])
        print(le.classes_[log_best_n[0][3]])
        print(le.classes_[log_best_n[0][2]])
        print(le.classes_[log_best_n[0][1]])
        print(le.classes_[log_best_n[0][0]])


    return render_template("index.html", log_predResult = log_predResult, searchTerms = searchTerms, log_probs=[log_probs], entered=entered)

@app.route("/rawdata/")
def rawdata():
    wine_data = pd.read_sql('wines', con=engine).to_dict("records")
    return jsonify(wine_data)

@app.route("/data/")
def data():
    return render_template("data.html")

@app.route("/stopWords/")
def stopWords():
    stop_words = pd.read_sql('stopWords', con=engine).to_dict("records")
    return jsonify(stop_words)

@app.route("/storedFeedback/", methods=['GET'])
def getFeedback():
    if request.method == 'GET':
        totalList = json.loads(request.args.get('totalList[]'))
        return jsonify(totalListi)

@app.route("/methodology/")
def methodology():
    return render_template("methodology.html")

@app.route("/data/logistic")
def logistic():
    return render_template("logistic.html")

@app.route("/data/svm")
def svm():
    return render_template("svm.html")

@app.route("/data/nlp")
def nlp():
    nlpImg = "img/nlp_accuracy.jpg"
    return render_template("nlp.html", nlpImg = nlpImg)

@app.route("/data/randomforest")
def randomforest():
    return render_template("randomforest.html")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
