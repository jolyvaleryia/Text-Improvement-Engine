import warnings
warnings.filterwarnings('ignore')

import nltk
import re
import string
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
import tensorflow_hub as hub
import numpy as np

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')


def text_transform(text_to_transform):
    transformed_text = text_to_transform.translate(str.maketrans('', '', string.punctuation)).split()
    return transformed_text


def delete_stop_words(words_to_transform):
    english_stopwords = stopwords.words('english')
    transformed_words = []
    for word in words_to_transform:
        word = word.lower()
        if word not in english_stopwords:
            transformed_words.append(word)
    return transformed_words


def phrases_transform(standard_phrases_to_transform):
    transformed_standard_phrases = re.split('\d+', str(standard_phrases_to_transform))[1:]

    for index in range(len(transformed_standard_phrases)):
        transformed_standard_phrases[index] = transformed_standard_phrases[index].strip().lower()
    return transformed_standard_phrases


def embedding(phrase):
    url1 = "https://tfhub.dev/google/elmo/3"
    url2 = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
    url3 = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.KerasLayer(url3)
    tensor = tf.constant([phrase])
    phrase_embeddings = np.asarray(embed(tensor))[0].reshape((1, -1))
    return phrase_embeddings


def calculate_similarity(words_list, phrases):
    for index in range(len(words_list)-1):
        for phrase in phrases:
            two_words = " ".join((words_list[index], words_list[index+1]))
            one_word = words_list[index]
            phrase_embed = embedding(phrase)
            for word in one_word, two_words:
                similarity = cosine_similarity(embedding(word), phrase_embed)
                if similarity > 0.65:
                    print("Original phrase:", word, ', ',
                          "Recommended replacement:", phrase, ', ',
                          "Similarity score:", similarity[0][0])


if __name__ == '__main__':
    with open('data/sample_text.txt') as file:
        text = file.read()
    words_list = text_transform(text)
    words_list = delete_stop_words(words_list)

    standard_phrases = pd.read_csv('data/Standardised terms.csv', index_col=None)
    standard_phrases = phrases_transform(standard_phrases)

    scientific_terms = pd.read_excel('data/CDISC Glossary.xls', sheet_name='Glossary Terminology 2023-12-15')
    scientific_terms = scientific_terms['CDISC Submission Value'].tolist()

    print('Standard_phrases:')
    calculate_similarity(words_list, standard_phrases)

    print('Scientific_terms:')
    calculate_similarity(words_list, scientific_terms)
