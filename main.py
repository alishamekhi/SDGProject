import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files
# Load Data
def load_data(data_dir, label):
    sequences = []
    for filepath in list_all_files(data_dir):
        with open(filepath, 'r') as file:
            seq = file.read().strip().split()
            sequences.append(' '.join(seq))  # Join system calls with space for CountVectorizer
    labels = [label] * len(sequences)
    return sequences, labels


normal_sequences, normal_labels = load_data('ADFA-LD/Training_Data_Master/', 0)
attack_sequences, attack_labels = load_data('ADFA-LD/Attack_Data_Master/', 1)

# Combine Data
all_sequences = normal_sequences + attack_sequences
all_labels = normal_labels + attack_labels

# Encode System Calls
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_sequences)
encoded_sequences = tokenizer.texts_to_sequences(all_sequences)

# Pad Sequences
max_length = 500  # Adjust based on your data distribution
X = pad_sequences(encoded_sequences, maxlen=max_length, padding='post')
y = np.array(all_labels)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
