import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load and prepare the dataset
problems_df = pd.read_csv('datasets/problems.csv')
solutions_df = pd.read_csv('datasets/solutions.csv')
merged_df = pd.merge(problems_df, solutions_df, on='id')
desc = merged_df['description']
ans = merged_df['answer']

# define model parameters
max_sequence_length = 100
vocab_size = 10000
embedding_dim = 256
batch_size = 64
epochs = 5

# tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(desc)
word_index = tokenizer.word_index

# convert text to sequences and pad them
X_seq = tokenizer.texts_to_sequences(desc)
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length, padding='post')

# prepare output data and pad it
y_seq = tokenizer.texts_to_sequences(ans)
y_padded = pad_sequences(y_seq, maxlen=max_sequence_length, padding='post')

# shift the padded sequences by one timestep for the decoder input
y_seq_shifted = np.roll(y_padded, -1, axis=1)

# convert output to one-hot encoding for the target
y_one_hot = tf.keras.utils.to_categorical(y_padded, num_classes=vocab_size)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_one_hot, test_size=0.2, random_state=42)
y_input_train, y_input_test, _, _ = train_test_split(y_seq_shifted, y_padded, test_size=0.2, random_state=42)

# build the model
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(128, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_sequence_length,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# training the model
model.fit([X_train, y_input_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# evaluate the model
loss = model.evaluate([X_test, y_input_test], y_test)
print(f'Test loss: {loss}')

# prediction example
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

def decode_sequence(sequence):
    return ' '.join([reverse_word_index.get(token, '?') for token in sequence])

new_problem = 'Write a program that asks the user for their name and greets them with their name.'
new_problem_seq = tokenizer.texts_to_sequences([new_problem])
new_problem_padded = pad_sequences(new_problem_seq, maxlen=max_sequence_length, padding='post')

# generate prediction
predicted_answer_seq = model.predict([new_problem_padded, np.zeros((1, max_sequence_length))])
print("Raw predictions:", predicted_answer_seq[0].argmax(axis=-1))

# convert the most likely token at each timestep to words
predicted_answer = decode_sequence(np.argmax(predicted_answer_seq[0], axis=-1))
print('Predicted Answer:', predicted_answer)
