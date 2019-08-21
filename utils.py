"""Just utilitary functions and classes."""

import hashlib
import itertools
import os
import re
import string
import xml.etree.ElementTree as ET

from gensim.models import KeyedVectors
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from pyemd import emd


class GradeXML2DataFrame:

    def __init__(self, xml_data):
        self.root = ET.parse(xml_data).getroot()
    
    def extract_label(self, label_string):
        """Function that returns the class Id from original label string.
        0: correct
        1: correct_but_incomplete
        2: contradictory
        3: incorrect

        Args:
            label_string (String): original label in XML file

        Returns:
            class_id (Int): target class Id (0, 1, 2 or 3)
        """
        
        # Create list of all labels
        label_list = np.array(label_string.split('|'))
        # Keep only the label with flag (1)
        class_id = [i for i, s in enumerate(label_list) if "(1)" in s][0]
        return class_id

    def parse_instance(self, instance):
        """Create a dictionary of instance attributes and their values."""

        info = {}
        info['instance_id'] = int(instance.attrib['ID'])
        for elem in instance:
            if elem.tag == 'MetaInfo':
                info['student_id'] = elem.attrib['StudentID']
                info['task_id'] = elem.attrib['TaskID']
            if elem.tag == 'Annotation':
                info['label'] = self.extract_label(elem.attrib['Label'])
            if elem.tag == 'ProblemDescription':
                info['problem_description'] = elem.text
            if elem.tag == 'Question':
                info['question'] = elem.text
            if elem.tag == 'Answer':
                info['answer'] = elem.text
            if elem.tag == 'ReferenceAnswers':
                ra = re.sub('\n', ' ', elem.text).strip()
                info['reference_answers'] = ra
        return info
    
    def todf(self):
        """Create a dataframe containing all instances and their details."""

        instances = []
        for instance in self.root:
            info = self.parse_instance(instance)   
            instances.append(info) 
        columns = [
            'instance_id', 'student_id', 'task_id', 'problem_description',
            'question', 'answer', 'reference_answers', 'label'
            ]
        return pd.DataFrame(instances, columns=columns)


def get_hash(text):
    """Generate a hashkey of text using the MD5 algorithm."""

    punctuation = string.punctuation
    punctuation = punctuation + "’" + "“" + "?" + "‘"
    text = [c if c not in punctuation else ' ' for c in text]
    text = ''.join(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = text.strip()
    return hashlib.md5(text.encode()).hexdigest()[:10]


def get_reference_answers(reference_answers):
    """Create a cleaned list of reference answers.

    Args:
        reference_answers (String): Original reference answers in XML file

    Returns:
        ra_list (list): reference answers separated in a list
    """

    # Split on new line character
    ra_list = reference_answers.split(':') 
    # Delete unecessary answer numbers
    ra_list = [re.sub('[0-9]+', '', ra).strip() for ra in ra_list] 
    # Remove empty elements
    ra_list = [ra for ra in ra_list if ra != '']
    return ra_list


class Featurizer:

    def __init__(self, embedding_file):

        if not os.path.exists(embedding_file):
            raise IOError("Embeddings file does not exist: %s" %embedding_file)

        punctuation = string.punctuation
        punctuation = punctuation + "’" + "“" + "?" + "‘"
        self.punctuation = punctuation
        print('INFO: Loading word vectors...')
        self.word2vec = KeyedVectors.load_word2vec_format(
            embedding_file,
            binary=True)

        print('INFO: Done! Using %s word vectors from pre-trained word2vec.' \
            %len(self.word2vec.vocab))
    
    def remove_punc(self, phrase):
        """Remove punctuation, changes to lower case, strips leading,
        middle and trailing spaces."""

        nopunc = [c if c not in self.punctuation else ' ' for c in phrase]
        nopunc = ''.join(nopunc)
        nopunc = re.sub(r'\s+', ' ', nopunc)
        nopunc = nopunc.lower()
        nopunc = nopunc.strip()
        return nopunc

    def tokenize(self, phrase):
        """Produce a list of tokens from a character string."""

        tokenized = phrase.split()
        return tokenized

    def filter_vocab(self, tokens):
        """Remove all tokens that are not in the pre-learned vocabulary."""
        
        tokens = [token for token in tokens if token in self.word2vec.vocab]
        return tokens

    def preprocess(self, phrase):
        """Create a list of word tokens to represent a phrase."""

        return self.filter_vocab(self.tokenize(self.remove_punc(phrase)))

    def tokens2vec(self, tokens):
        """Generate the mean embedding of a list of tokens."""
        
        emb = np.zeros(300)
        if len(tokens) > 0:
            nb_tokens = 0
            for token in tokens:
                emb += self.word2vec.word_vec(token)
                nb_tokens += 1
            emb /= nb_tokens
        return emb

    def doc2vec(self, phrase):
        """Generate the phrase embedding as a mean of its tokens."""
        
        tokens = self.preprocess(phrase)
        emb = self.tokens2vec(tokens)
        return emb

    def cossim_from_emb(self, emb1, emb2):
        """Compute the cosine similarity between two numpy vectors."""
        
        if norm(emb1)==0 or norm(emb2)==0:
            sim = 0
        else:
            sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return sim

    def l2_dist(self, emb1, emb2):
        """Compute the l2 distance between two numpy vectors."""
        
        dist = norm(emb1 - emb2)
        return dist

    def cossim_from_phrase(self, phrase1, phrase2):
        """Compute the cosine similarity between two phrases."""

        emb1 = self.doc2vec(phrase1)
        emb2 = self.doc2vec(phrase2)
        sim = self.cossim_from_emb(emb1, emb2)
        return sim

    def asym_diff(self, phrase1, phrase2):
        """Compute a distance between (token2 - token1) and token1
        Where tokens1 and tokens2 are from phrase1 and from phrase2.
        This is an asymmetric distance.
        If tokens1 is from student answer and tokens2 is from reference answer,
        the distance when high is supposed to detect correct_but_incomplete answers.
        If tokens1 is from reference answer and tokens2 is from student answer,
        the distance when high is supposed to detect contradictory answers.
        """

        # Tokens in phrase 1
        tokens1 = self.preprocess(phrase1)
        # Tokens in phrase 2
        tokens2 = self.preprocess(phrase2)
        # Tokens in 2 but not in 1
        diff = [token for token in tokens2 if token not in tokens1]
        
        if len(tokens2)==0:
            dist = 0
        else:
            dist = len(diff) / len(tokens2)
        return dist

    def word_match(self, phrase1, phrase2):
        """Computes the ratio of tokens in common."""
        # Tokens in phrase 1
        tokens1 = self.preprocess(phrase1)
        # Tokens in phrase 2
        tokens2 = self.preprocess(phrase2)
        # Tokens in 2 but not in 1
        common = [token for token in tokens2 if token in tokens1]
        
        if len(tokens1)==0 and len(tokens2)==0:
            match = 0
        else:
            match = len(common) / (len(tokens1) + len(tokens2))
        return match

    def wmdist(self, phrase1, phrase2):
        """Computes the world mover's distance between the two phrases."""

        # Tokens in phrase 1
        tokens1 = self.preprocess(phrase1)
        # Tokens in phrase 2
        tokens2 = self.preprocess(phrase2)
        
        if len(tokens1)==0 or len(tokens2)==0:
            dist = 1
        else:
            dist = self.word2vec.wmdistance(tokens1, tokens2)
        return dist


def plot_confusion_matrix(cm, classes, normalize=False,
                            title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_model(model_file):
    """Load a pretained model from pickel file."""

    if not os.path.exists(model_file):
            raise IOError("Model file does not exist: %s" %model_file)
    model = joblib.load(model_file)
    print("INFO: Model %s was loaded successfully." \
        %os.path.basename(model_file))
    return model
