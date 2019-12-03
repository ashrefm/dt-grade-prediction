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

"""Prepare train, test and landmark embeddings."""

import os
import re
from time import time

import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import GradeXML2DataFrame
from utils import get_hash
from utils import get_reference_answers
from utils import Featurizer


def train_test_landmark_split(xml_data):
    """Prepare train, dev and test data from original XML into csv format."""

    # Preprocess all data into a Pandas dataframe
    xml2df = GradeXML2DataFrame(xml_data)
    data = xml2df.todf()
    # Modeling/Test split
    remaining, test = train_test_split(data, test_size=0.2, random_state=22)
    # Train/Dev split
    train, landmarks = train_test_split(
        remaining,
        test_size=0.125,
        random_state=22)
    # Save modeling and test data
    if not os.path.exists('munge'):
        os.makedirs('munge')
    data.to_csv(os.path.join('munge', 'grade_data.csv'), index=False)
    train.to_csv(os.path.join('munge', 'train.csv'), index=False)
    landmarks.to_csv(os.path.join('munge', 'landmarks.csv'), index=False)
    test.to_csv(os.path.join('munge', 'test.csv'), index=False)
    print("INFO: Train, test and landmarks data were created successfully "
    "in munge directory.")


def landmarks(reference_landmarks, student_landmarks):
    """Create a text file with word2vec embeddings of student
    and reference answers and their labels.

    Args:
        reference_landmarks: csv to extract reference landmarks
        student_landmarks: csv to extract student landmarks
    """

    # Init timer
    start = time()

    # Read a dataset containing all reference answers
    ra_data = pd.read_csv(reference_landmarks) 
    # Create hash keys for problem description and question
    ra_data['pd_hash'] = ra_data['problem_description'].apply(get_hash)
    ra_data['qu_hash'] = ra_data['question'].apply(get_hash)
    # Create a dataframe of reference answers one per row
    ra_data['ra_list'] = ra_data['reference_answers']\
        .apply(get_reference_answers)
    landmarks_ra = ra_data[['pd_hash', 'qu_hash', 'label', 'ra_list']]
    landmarks_ra = landmarks_ra.explode('ra_list')
    landmarks_ra = landmarks_ra.rename(columns={'ra_list':'answer'})
    landmarks_ra['label'] = 0 # these are possible correct answers (class 0)
    landmarks_ra = landmarks_ra.drop_duplicates()
    print("INFO: Found {} distinct reference landmark answers."\
        .format(len(landmarks_ra)))

    # Create a dataframe of student answers
    sa_data = pd.read_csv(student_landmarks)
    # Create hash keys for problem description and question
    sa_data['pd_hash'] = sa_data['problem_description'].apply(get_hash)
    sa_data['qu_hash'] = sa_data['question'].apply(get_hash)
    landmarks_sa = sa_data[['pd_hash', 'qu_hash', 'label', 'answer']]
    landmarks_sa = landmarks_sa.drop_duplicates()
    print("INFO: Found {} distinct student landmark answers."\
        .format(len(landmarks_sa)))

    # Create the landmarks dataframe with distinct answers and their labels
    landmarks = landmarks_ra.append(landmarks_sa).drop_duplicates()

    # Create a featurizer object that converts a phrase into embedding vector
    emb_file = os.path.join('data', 'GoogleNews-vectors-negative300.bin')
    featurizer = Featurizer(emb_file)

    # Save the embeddings and labels to disk
    n_landmarks = 0
    with open(os.path.join('munge', 'landmarks.txt'), 'w') as f:
        f.write('pd_hash\tqu_hash\tlabel\tanswer\tembedding\n')
        for i in range(len(landmarks)):
            pd_hash = landmarks.iloc[i]['pd_hash']
            qu_hash = landmarks.iloc[i]['qu_hash']
            label = landmarks.iloc[i]['label']
            answer = landmarks.iloc[i]['answer']
            emb = featurizer.doc2vec(landmarks.iloc[i]['answer'])
            emb_txt = ','.join(map(str, emb))
            if norm(emb) != 0:
                n_landmarks += 1
                f.write("%s\t%s\t%s\t%s\t%s\n"\
                    %(pd_hash, qu_hash, label, answer, emb_txt))
    print('INFO: Generating landmark embeddings took %.2f seconds.' \
        %(time() - start))
    print("INFO: Found {} non zero landmarks in total.".format(n_landmarks))


def main():
    train_test_landmark_split(os.path.join('data', 'grade_data.xml'))
    landmarks(
        os.path.join('munge',  'grade_data.csv'),
        os.path.join('munge',  'landmarks.csv')
    )
    

if __name__ == "__main__":
    main()