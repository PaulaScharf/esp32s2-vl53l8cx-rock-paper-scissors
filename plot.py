import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, :].values  # Assuming first column is a header or label
    return data

# Function to save frames as images
def save_frames(data, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    for i, row in enumerate(data):
        frame = row.reshape(8, 8)  # Reshape to 8x8
        plt.imshow(frame, cmap='viridis', vmin=0, vmax=400)  # Adjust color scale to match value range
        plt.colorbar(label='Intensity')
        plt.title(f"{prefix} - frame {i + 1}")
        plt.axis('off')

        output_path = os.path.join(output_folder, f"plots\{prefix}_frame_{i + 1}.png")
        plt.savefig(output_path)
        plt.close()

# Paths to your CSV files
file_paths = ["data\paper2.csv", "data\\rock.csv", "data\scissors.csv"]

# Loop through each file and save frames
for file_path in file_paths:
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract file name without extension
    output_folder = os.path.dirname(file_path)  # Use the same folder as the CSV file

    print(f"Processing {file_path}...")
    data = load_data(file_path)
    save_frames(data, output_folder, prefix=file_name)
    print(f"Saved frames from {file_path} to {output_folder}")

print("All frames processed and saved.")
