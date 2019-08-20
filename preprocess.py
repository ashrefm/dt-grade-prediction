"""Prepare train, dev, test and landmark embeddings."""

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


def train_dev_test_split(xml_data):
    """Prepare train, dev and test data from original XML into csv format."""

    # Preprocess all data into a Pandas dataframe
    xml2df = GradeXML2DataFrame(xml_data)
    data = xml2df.todf()
    # Modeling/Test split
    modeling, test = train_test_split(data, test_size=0.2, random_state=22)
    # Train/Dev split
    train, dev = train_test_split(modeling, test_size=0.75, random_state=22)
    # Save modeling and test data
    if not os.path.exists('munge'):
        os.makedirs('munge')
    data.to_csv(os.path.join('munge', 'grade_data.csv'), index=False)
    modeling.to_csv(os.path.join('munge', 'modeling.csv'), index=False)
    train.to_csv(os.path.join('munge', 'train.csv'), index=False)
    dev.to_csv(os.path.join('munge', 'dev.csv'), index=False)
    test.to_csv(os.path.join('munge', 'test.csv'), index=False)
    print("Train, dev and test data were saved successfully "
    "in munge directory.")


def landmarks(csv_data):
    """Create a text file with word2vec embeddings of student
    and reference answers and their labels.

    Args:
        csv_data: path to csv file containing the instances
    """

    # Read training set
    data = pd.read_csv(csv_data)

    # Create hash keys for problem description and question
    data['pd_hash'] = data['problem_description'].apply(get_hash)
    data['qu_hash'] = data['question'].apply(get_hash)

    # Create a dataframe of reference answers one per row
    data['ra_list'] = data['reference_answers'].apply(get_reference_answers)
    landmarks_ra = data[['pd_hash', 'qu_hash', 'label', 'ra_list']]
    landmarks_ra = landmarks_ra.explode('ra_list')
    landmarks_ra = landmarks_ra.rename(columns={'ra_list':'answer'})
    landmarks_ra['label'] = 0 # these are possible correct answers (class 0)
    landmarks_ra = landmarks_ra.drop_duplicates()
    print("Found {} distinct reference landmark answers."\
        .format(len(landmarks_ra)))

    # Create a dataframe of student answers
    landmarks_a = data[['pd_hash', 'qu_hash', 'label', 'answer']]
    landmarks_a = landmarks_a.drop_duplicates()
    print("Found {} distinct student landmark answers."\
        .format(len(landmarks_a)))

    # Create the landmarks dataframe with distinct answers and their labels
    landmarks = landmarks_ra.append(landmarks_a).drop_duplicates()
    landmarks = landmarks_ra.drop_duplicates()

    # Create a featurizer object that converts a phrase into embedding vector
    emb_file = os.path.join('munge', 'GoogleNews-vectors-negative300.bin')
    featurizer = Featurizer(emb_file)

    # Save the embeddings and labels to disk
    start = time()
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
    print("Generating landmark embeddings took %.2f seconds."\
        %(time() - start))
    print("Found {} non zero landmarks in total.".format(n_landmarks))


def main():
    train_dev_test_split(os.path.join('data', 'grade_data.xml'))
    landmarks(os.path.join('munge',  'modeling.csv'))
    

if __name__ == "__main__":
    main()