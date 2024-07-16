import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Function to list all files in a directory
def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

# Function to load data
def load_data(data_dir):
    sequences = []
    for filepath in list_all_files(data_dir):
        with open(filepath, 'r') as file:
            seq = file.read().strip().split()
            sequences.append(seq)
    return sequences

# Load data from ADFA-LD dataset
training_data_dir = 'ADFA-LD/Training_Data_Master/'
attack_data_dir = 'ADFA-LD/Attack_Data_Master/'

normal_sequences = load_data(training_data_dir)
attack_sequences = load_data(attack_data_dir)

# Combine data and labels
all_sequences = normal_sequences + attack_sequences
all_labels = [0] * len(normal_sequences) + [1] * len(attack_sequences)

# Tokenize sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_sequences)
encoded_sequences = tokenizer.texts_to_sequences(all_sequences)

# Pad sequences
max_length = 500
X = pad_sequences(encoded_sequences, maxlen=max_length, padding='post')
y = np.array(all_labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build LSTM model for classification
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate model on real data
y_pred_real = (model.predict(X_test) > 0.5).astype("int32")
print("Classification Report on Real Data:")
report_real = classification_report(y_test, y_pred_real, output_dict=True)
print(classification_report(y_test, y_pred_real))

# Define GAN architecture
latent_dim = 100

# Generator model
generator = Sequential([
    Dense(128, input_shape=(latent_dim,), activation='relu'),
    Dense(max_length * (len(tokenizer.word_index) + 1), activation='sigmoid'),
    Reshape((max_length, len(tokenizer.word_index) + 1))
])

# Compile Discriminator
discriminator = Sequential([
    Input(shape=(max_length, len(tokenizer.word_index) + 1)),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

# Compile Discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
gan_input = Input(shape=(latent_dim,))
gen_output = generator(gan_input)
gan_output = discriminator(gen_output)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Train GAN
num_samples = len(attack_sequences)  # Number of samples to generate
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_sequences = generator.predict(noise)
synthetic_sequences_padded = pad_sequences(synthetic_sequences.argmax(axis=2), maxlen=max_length, padding='post')

# Combine normal and synthetic attack sequences for evaluation
X_combined = np.concatenate((X, synthetic_sequences_padded))
y_combined = np.concatenate((y, np.ones(num_samples)))

# Evaluate model on combined data
y_pred_combined = (model.predict(X_combined) > 0.5).astype("int32")
print("\nClassification Report on Combined Data:")
report_combined = classification_report(y_combined, y_pred_combined, output_dict=True)
print(classification_report(y_combined, y_pred_combined))

# Train the classifier with synthetic attack data and normal data
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# Re-train model
model.fit(X_train_combined, y_train_combined, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate model on real data after training with synthetic data
y_pred_real_after_synthetic = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report on Real Data after Training with Synthetic Data:")
report_real_after_synthetic = classification_report(y_test, y_pred_real_after_synthetic, output_dict=True)
print(classification_report(y_test, y_pred_real_after_synthetic))

# Function to plot the classification report
def plot_classification_report(report, title):
    df = pd.DataFrame(report).transpose()
    df = df.drop(columns=['support'])
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.show()

# Plot the classification reports
plot_classification_report(report_real, "Classification Report on Real Data")
plot_classification_report(report_combined, "Classification Report on Combined Data")
plot_classification_report(report_real_after_synthetic, "Classification Report on Real Data after Training with Synthetic Data")
