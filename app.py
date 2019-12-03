# MIT License
#
# Copyright (c) 2019 Mohamed-Achref MAIZA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

"""Flask App to demonstrate the ML inference system."""

import json
import os

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from pyemd import emd
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for

from utils import get_hash, load_model
from utils import Featurizer


app = Flask(__name__)


def get_features(landmarks, pd_hash, qu_hash, answer):
    """"Create all features to represent an observation.
    
    Args:
        landmarks (dataframe): landmark instances from preprocessing
        pd_hash (string) : hashkey of problem description
        qu_hash (string) : hashkey of question
        answer (string) : student response to question
    
    Returns:
        features : a numpy vector with float numbers
    """
        
    # Get observation values
    emb = featurizer.doc2vec(answer)
        
    # Landmark of different question will get zero similarity (default)
    qu_land = landmarks.copy()
    qu_land['similarity'] = 0 # Compute cosine similarity with landmark
    qu_land['asym_diff_left'] = 0 # asymmetric difference between answer and landmark
    qu_land['asym_diff_right'] = 0 # asymmetric diffence between landmark and answer
    qu_land['word_match'] = 0 # word match between landmark and answer
    #qu_land['wmdist'] = 0 # world mover's distance between landmark and answer

    # Get index of landmarks with same problem  and question 
    idx = qu_land[(qu_land['pd_hash']==pd_hash)\
         & (qu_land['qu_hash']==qu_hash)].index
    question_found = 1 if len(idx)>0 else 0

    # Compute similarity when embedding is not zero and landmark from same question
    if norm(emb)!=0:
        # Compute the direct similarity with these landmarks
        qu_land.loc[idx, 'similarity'] = qu_land.loc[idx, 'embedding']\
            .apply(lambda x : featurizer.cossim_from_emb(emb, np.array(x)))
        # Compute the asymmetric difference between answer and landmark
        qu_land.loc[idx, 'asym_diff_left'] = qu_land.loc[idx, 'answer']\
            .apply(lambda x : featurizer.asym_diff(answer, x))
        # Compute the asymmetric difference between landmark and answer
        qu_land.loc[idx, 'asym_diff_right'] = qu_land.loc[idx, 'answer']\
            .apply(lambda x : featurizer.asym_diff(x, answer))
        # Compute the word match ratio between answer and landmark
        qu_land.loc[idx, 'word_match'] = qu_land.loc[idx, 'answer']\
            .apply(lambda x : featurizer.word_match(answer, x))
        # Compute the world mover's distance between answer and landmark
        #qu_land.loc[idx, 'wmdist'] = qu_land.loc[idx, 'answer']\
        # .apply(lambda x : featurizer.wmdist(obs['answer'], x))

    # Features will be all similarity measures to landmarks    
    features = np.concatenate((qu_land['similarity'],
                               qu_land['asym_diff_left'],
                               qu_land['asym_diff_right'],
                               qu_land['word_match']))
     
    # Add feature to indicate wether or not observation embedding was zero
    void_answer = 0
    if norm(emb)==0:
        features = np.append(1, features)
        void_answer = 1
    else:
        features = np.append(0, features)
    
    return features, question_found, void_answer


def get_prediction(landmarks, instance):
    """Compute the prediction from instance features
    
    Args:
        landmarks (dataframe): landmark instances from preprocessing
        instance (dict) : contains intance data (pb_hash, qu_hash, answer)

    Returns: 
        prediction (string) : e.g correct, incorrect,..
    """
    
    pd_hash = instance['pd_hash']
    qu_hash = instance['qu_hash']
    answer = instance['answer']
    features, question_found, void_answer = get_features(
        landmarks,
        pd_hash,
        qu_hash,
        answer)
    if not question_found: return "Unknown"
    if void_answer: return "Incorrect"
    features = np.reshape(features, (1, len(features)))
    class_id = model.predict(features)[0]
    class_dict = {
        0:'Correct', 
        1:'Correct but incomplete',
        2:'Contradictory',
        3:'Incorrect'
    }
    prediction = class_dict[class_id]
    return prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form')
def input_form():
    return render_template('form.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = json.loads(request.data.decode())
    mandatory_items = ['problem', 'question', 'answer']
    for item in mandatory_items:
        if item not in data.keys():
            return jsonify({'result': 'Fill in all text fields.'})
        if item == '':
            return jsonify({'result': 'Empty field: %s' %item})
    instance = {}
    instance['pd_hash'] = get_hash(data['problem'])
    instance['qu_hash'] = get_hash(data['question'])
    instance['answer'] = data['answer']
    prediction = get_prediction(landmarks, instance)
    return jsonify({'result': prediction})

    
if __name__ == "__main__":

    print("INFO: Starting DT-Grade Prediction App...")
    
    #  Read landmarks
    landmarks = pd.read_csv(os.path.join('munge', 'landmarks.txt'), sep="\t")
    landmarks['embedding'] = landmarks['embedding']\
        .apply(lambda x : list(map(float, x.split(','))))
    if len(landmarks) == 0:
        raise Exception("Landmarks file is empty.")
    else:
        print("INFO: Found %d landmark instances." %len(landmarks))

    # Create a featurizer object that converts a phrase into embedding
    # vector using pre-trained word2vec
    emb_file = os.path.join('data', 'GoogleNews-vectors-negative300.bin')
    featurizer = Featurizer(emb_file)

    # Load model
    model = load_model(os.path.join('model', 'multinomial_lr.pkl'))
    app.run(debug=True)