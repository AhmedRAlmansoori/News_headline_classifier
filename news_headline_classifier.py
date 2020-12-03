import tensorflow as tf
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sentences = []
labels = []

#importing the data set from file
with open(r'bbc-text.csv') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter = ',')
    next(csv_reader)
    for line in csv_reader:
        labels.append(line[0])
        sentences.append(line[1])

print('Retrieved {} labels from file.'.format(len(labels)))
print('Retrieved {} sentences from file.'.format(len(sentences)))

training_data_size = 1750

training_sentences = sentences[:training_data_size]
training_labels    = labels[:training_data_size]
testing_sentences  = sentences[training_data_size:]
testing_labels     = labels[training_data_size:]

print("data split into {} for training and {} for testing".format(training_data_size,len(sentences)-training_data_size))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(sentences),
                                                  oov_token='OOV')
label_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10)

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

padding_type = 'post'
training_sequence = tokenizer.texts_to_sequences(training_sentences)
training_padded =pad_sequences(training_sequence,
                              padding=padding_type)


label_tokenizer.fit_on_texts(training_labels)

label_word_index = label_tokenizer.word_index

print(len(label_tokenizer.word_index))

training_labels = label_tokenizer.texts_to_sequences(training_labels)
testing_labels = label_tokenizer.texts_to_sequences(testing_labels)

testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
testing_padded =pad_sequences(testing_sequence,
                              padding=padding_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(sentences),16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6,  activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer ='rmsprop',metrics =['accuracy'])

model.summary()
print(len(testing_labels))

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

history = model.fit(training_padded,training_labels,
                    batch_size=25,
                    epochs=35,
                    validation_data=(testing_padded,testing_labels),
                    verbose=2)

test_cases = [
    'Boeing 737 Max sees first firm order since crashes',
    'Google fired employees for union activity, says US agency',
    'The UAE and Israels whirlwind honeymoon has gone beyond normalization',
    'Why Donald Trump keeps outperforming the polls',
    'Self-driving robotaxis are taking off in China',
    'Why some food brands want you to know their climate impact']

test_labels=['business',
             'tech',
             'politics',
             'politics',
             'tech',
             'business']

test_sequences = tokenizer.texts_to_sequences(test_cases)

for i in range(0,len(test_sequences)):
    prediction = model.predict(test_sequences[i])
    predicted_label = testing_labels[np.argmax(prediction)]

    print(np.argmax(prediction))
    print('Model Predicted : {}'.format(test_cases[i]))
    print('Actual Category :', test_labels[i])
    for i in label_word_index:
        if label_word_index[i]==predicted_label:
            print('Predicated Category : {}'.format(i))


