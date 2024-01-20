import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed

# data
problems_df = pd.read_csv('datasets/problems.csv')
solutions_df = pd.read_csv('datasets/solutions.csv')

# merge datasets based on the 'id' column
merged_df = pd.merge(problems_df, solutions_df, on='id', how='inner')

# drop rows with missing values in the 'description' or 'answer' columns
merged_df = merged_df.dropna(subset=['description', 'answer'])

# tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(merged_df['description'])

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocabulary Size:", vocab_size)

input_sequences = tokenizer.texts_to_sequences(merged_df['description'])
target_sequences = tokenizer.texts_to_sequences(merged_df['answer'])

# pad both input and target sequences to a fixed length
max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
padded_target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

print("Input Sequences Shape:", padded_input_sequences.shape)
print("Target Sequences Shape (Shifted):", padded_target_sequences.shape)

# define and train the s2s model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len, mask_zero=True),
    LSTM(100, return_sequences=True),
    LSTM(100, return_sequences=True),  # Use return_sequences for each LSTM layer
    TimeDistributed(Dense(vocab_size, activation='softmax'))  # TimeDistributed layer for sequence generation
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Input Sequences Shape:", padded_input_sequences.shape)
print("Target Sequences Shape (Shifted):", padded_target_sequences.shape)

model.fit(padded_input_sequences, padded_target_sequences, epochs=5, batch_size=32)

# generate predictions
example_input_sequence = padded_input_sequences[0:1]
predicted_sequence = model.predict(example_input_sequence)

# decode the predicted sequence back to text
predicted_text = ' '.join([tokenizer.index_word[idx] for idx in predicted_sequence[0].argmax(axis=-1) if idx != 0])
print("Predicted Solution:", predicted_text)
