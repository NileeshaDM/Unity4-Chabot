import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random

#nltk.download('stopwords')

# initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# sw = stopwords.words('english')

# initialize empty lists and variables
words = []
classes = []
documents = []
ignore = ['?', '!']

# get the intents
data = open('intents.json').read()
intents = json.loads(data)

# preprocess and collect data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        # words.extend([i for i in w if i not in sw])
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add the intent to the list of classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# dump the words and classes
with open('words.pkl', 'wb') as words_file:
    pickle.dump(words, words_file)
with open('classes.pkl', 'wb') as classes_file:
    pickle.dump(classes, classes_file)

# create the training data
training = []

# create an empty array for output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = [0] * len(words)
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1 if word match found in the current pattern
    for w in pattern_words:
        if w in words:
            bag[words.index(w)] = 1

    # output is '0' for each tag and '1' for the current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)

# separate the input features (X) and the target labels (Y)
X = np.array([item[0] for item in training])
Y = np.array([item[1] for item in training])

# create the model
model = Sequential()
model.add(Dense(128, input_dim=len(X[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y[0]), activation='softmax'))

# compile the model
adam_optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(X, Y, epochs=200, batch_size=5, verbose=1)
model.save('foodbot_model.h5', hist)

