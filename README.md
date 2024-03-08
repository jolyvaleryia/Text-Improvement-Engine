# Texte Improvement Engine

## Contents of this file

 - Introduction
 - Requirements
 - Installation
 - Description of functions
 - Maintainers

## Introduction

It is a Texte Improvement Engine that analyses a text using a pre-trained language model, finds phrases in the input text that are semantically similar to any of the standardised phrases and suggests the possible recommended replacement based on the cosine similarity.

## Requirements

- Python 3.9.13
  
- Virtual Environment with installed libraries pandas, tensorflow, tensorflow_hub, scikit-learn and nltk.
  
- Pre-load a list of English stopwords.

## Installation

Download a project from Github and run main.py.

## Description of functions

- text_transform() - removes punctuation and splits text into a word list.

- delete_stop_words() - deletes stopwords from the list of words.

- phrases_transform() - downloads from the csv-file a list of standard_phrases and transforms them into a list.

- embedding() - creates phrase embeddings for a given phrase using 1 of the 3 models.

- calculate_similarity() - calculates cosine similarity for 2 phares of for a words and a phrase.

## Maintainers

Current maintainers:
- Valeryia Joly
