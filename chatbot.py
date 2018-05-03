# building a chatbot with deep NLP

# import libraries
import numpy as np
import tensorflow as tf
import re 
import time

###### data processing  ######

# import dataset
lines = open('movie_lines.txt').read().split('\n')
conversations = open('movie_conversations.txt').read().split('\n')

# create dictionary mapping each line with id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# create dictionary for conversations ids
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(","))

# get questions and answers seperately
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# first cleaning of text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"can ' t", "cannot", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"[^a-zA-Z\d\s:]", "", text)

    return text

# cleaning questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# cleaning answers:
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# create dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split(" "):
        if word not in word2count:
            word2count[word] = 1
        else: 
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split(" "):
        if word not in word2count:
            word2count[word] = 1
        else: 
            word2count[word] += 1        
        
# create two dictionaries to map answer words and question words to a unique integer
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
   
answerwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1

# add last tokens to dictionaries
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    answerwords2int[token] = len(answerwords2int) + 1        

# create inverse answerwords2int dictionary
answersint2words = {w_i: w for w, w_i in answerwords2int.items()}
        
        
        
        
