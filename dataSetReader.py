import pandas as pd
import numpy as np
from collections import defaultdict

# Load CSV file
data = pd.read_csv('Health_Data_Set.csv', header=None)

# Assuming: columns 0-3 are symptoms, column 4 is diagnosis
data = data.values
data = data[1:]

# Count occurrences of each unique (s1, s2, s3, s4, diagnosis)
counts = defaultdict(int)

for row in data:
    key = tuple(row)  # (s1, s2, s3, s4, diagnosis)
    counts[key] += 1

# Total number of records
total = len(data)

# Create full joint distribution as a 5D array (2x2x2x2x4)
joint_prob = np.zeros((2, 2, 2, 2, 4))

for key, count in counts.items():
    print(f"Key: {key}, Count: {count}")
    s1, s2, s3, s4, diag = key
    joint_prob[int(s1), int(s2), int(s3), int(s4), int(diag)] = count / total  # normalize
    # Grab the joint probabilities for all 4 diagnoses at this symptom config
    diagnosis_probs = joint_prob[int(s1), int(s2), int(s3), int(s4), :]

    # Normalize to get conditional probability P(diagnosis | symptoms)
    total_prob = np.sum(diagnosis_probs)
    if total_prob > 0:
        posterior = diagnosis_probs / total_prob
    else:
        posterior = np.zeros_like(diagnosis_probs)  # or handle zero-case with smoothing

    # Print it out
    for diag, prob in enumerate(posterior):
        print(f"P(diagnosis = {diag} | symptoms = [{int(s1)}, {int(s2)}, {int(s3)}, {int(s4)}]) = {prob:.3f}")
    print()  # Newline for readability

# Done — joint_prob is your full distribution
# Now you can use joint_prob for further analysis or modeling

print("Inference for P(diagnosis | s1=1):")

numerator = np.zeros(4) # 4 diagnoses

for s2 in [0, 1]:
    for s3 in [0, 1]:
        for s4 in [0, 1]:
            for diag in range(4):
                prob = joint_prob[1, s2, s3, s4, diag]
                numerator[diag] += prob

# Normalize the numerator to get P(diagnosis | s1=1)
denominator = np.sum(numerator)
if denominator > 0:
    posterior = numerator / denominator
else:
    posterior = np.zeros_like(numerator)
# Print the results
for diag, prob in enumerate(posterior):
    print(f"P(diagnosis = {diag} | s1 = 1) = {prob:.3f}")
# Done — posterior is your conditional distribution P(diagnosis | s1=1)
