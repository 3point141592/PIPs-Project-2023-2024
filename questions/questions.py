import nltk
import sys
import os
import string
import math
import pickle
from nltk.corpus import wordnet
import random

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

global FILE_MATCHES, SENTENCE_MATCHES, ALT_QUERYS
FILE_MATCHES = 1
SENTENCE_MATCHES = 1
ALT_QUERIES = 12

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    while query:
        
            # Determine top file matches according to TF-IDF
            filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

            # Extract sentences from top files
            sentences = dict()
            for filename in filenames:
                for passage in files[filename].split("\n"):
                    for sentence in nltk.sent_tokenize(passage):
                        tokens = tokenize(sentence)
                        if tokens:
                            sentences[sentence] = tokens

            # Compute IDF values across sentences
            idfs = compute_idfs(sentences)

            # Determine top sentence matches
            matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
            for match in matches:
                print(match)

        

            query = set(tokenize(input("Query: ")))


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    def read(file):
        with open(file, 'rb') as file:
            res = file.read().decode()
        return res
            
    return {file: read(os.path.join(directory, file)) for file in os.listdir(directory)}


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    return [word.lower() for word in words if not word.lower() in string.punctuation and not word.lower() in nltk.corpus.stopwords.words('english')]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for val in documents.values():
        words = words.union(set(val))

    res = {word: 0 for word in words}

    for ls in documents.values():
        for word in res:
            res[word] += ls.count(word)

    return {word: math.log(len(documents)) / res[word] for word in res}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_ratings = {file: 0 for file in files}

    for file in files:
        for word in query:
            tf_idf_ratings[file] += idfs[word] * files[file].count(word)

    return sorted(tf_idf_ratings, reverse=True, key=tf_idf_ratings.get)[:n]

def tf_idf(query, sentences, idfs, n):
    tf_idf_ratings = {sentence: 0 for sentence in sentences}
    for sentence in sentences:
        for word in query:
            tf_idf_ratings[sentence] += idfs[word] * sentences[sentence].count(word)

    return tf_idf_ratings

def synonyms(word):
    syns = wordnet.synsets(word)
    for syn in syns:
        for word in syn.lemmas():
            yield word.name().lower()

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    tf_idf_ = tf_idf(query, sentences, idfs, n)
    '''syns = {word: list(synonyms(word)) for word in query}
    alt_queries = []
    for i in range(ALT_QUERIES):
        l = random.choice(list(syns.keys()))
        new = random.choice(syns[l])
        q = query.copy()
        q.remove(l)
        q.add(new)
        alt_queries.append(q)

    for alt in alt_queries:
        try:
            ratings = tf_idf(alt, sentences, idfs, n)
            for x in tf_idf_:
                tf_idf_[x] += ratings[x]

        except:
            pass'''

    return sorted(tf_idf_, key=tf_idf_.get, reverse=True)[:n]


if __name__ == "__main__":
    main()
