import os
import matplotlib.pyplot as plt

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def compute_average_sequence_length(data_dir):
    total_sequences = 0
    total_length = 0

    # Iterate through files in the directory
    for filepath in list_all_files(data_dir):
        with open(filepath, 'r') as file:
            # Read the sequence (assuming each line is a sequence)
            sequence = file.read().strip().split()
            sequence_length = len(sequence)
            total_sequences += 1
            total_length += sequence_length

    # Calculate average sequence length
    if total_sequences > 0:
        average_length = total_length / total_sequences
    else:
        average_length = 0

    return average_length


def plot_sequence_length_distribution(data_dir, dataset_name):
    sequence_lengths = []

    # Iterate through files in the directory
    for filepath in list_all_files(data_dir):
        with open(filepath, 'r') as file:
            # Read the sequence (assuming each line is a sequence)
            sequence = file.read().strip().split()
            sequence_length = len(sequence)
            sequence_lengths.append(sequence_length)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Sequence Length Distribution - {dataset_name}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Sequences')
    plt.grid(True)
    plt.show()


# Specify the paths to Training_Data_Master and Attack_Data_Master directories
training_data_dir = 'ADFA-LD/Training_Data_Master'
attack_data_dir = 'ADFA-LD/Attack_Data_Master'

# Compute average sequence lengths
avg_length_training = compute_average_sequence_length(training_data_dir)
avg_length_attack = compute_average_sequence_length(attack_data_dir)

# Print average sequence lengths
print(f'Average Sequence Length - Training_Data_Master: {avg_length_training:.2f}')
print(f'Average Sequence Length - Attack_Data_Master: {avg_length_attack:.2f}')



# Plot distributions
plot_sequence_length_distribution(training_data_dir, 'Normal Sequences')
plot_sequence_length_distribution(attack_data_dir, 'Attack Sequences')
