#Jesus Ortega
#Omar Marquez
#Orion Wood
import pandas as pd
import numpy as np
from collections import defaultdict     # Automatically initializes new keys with a default value
from sklearn.preprocessing import LabelEncoder  # For converting string disease names to integers (or vice versa)

# Load dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Select and convert symptom columns
symptom_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
df_symptoms = df[symptom_cols + ["Disease"]].copy()
df_symptoms[symptom_cols] = df_symptoms[symptom_cols].applymap(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Encode disease labels
le_disease = LabelEncoder()
df_symptoms["Disease"] = le_disease.fit_transform(df_symptoms["Disease"])
unique_diseases = list(le_disease.classes_)
num_diseases = len(unique_diseases)

# Create joint probability table
joint_prob = np.zeros((2, 2, 2, 2, num_diseases))
counts = defaultdict(int)
for row in df_symptoms.values:
    key = tuple(row)
    counts[key] += 1

total = len(df_symptoms)
for key, count in counts.items():
    s1, s2, s3, s4, diag = key
    joint_prob[int(s1), int(s2), int(s3), int(s4), int(diag)] = count / total

# Inference function
def infer_from_joint(joint_prob, query_vector, num_diseases):
    assert query_vector.count(-2) == 1, "Exactly one query variable must be specified."
    query_index = query_vector.index(-2)

    result = {}
    domains = {
        0: [0, 1],
        1: [0, 1],
        2: [0, 1],
        3: [0, 1],
        4: list(range(num_diseases)),
    }

    for qval in domains[query_index]:
        total = 0.0
        for s1 in domains[0] if query_vector[0] < 0 else [query_vector[0]]:
            for s2 in domains[1] if query_vector[1] < 0 else [query_vector[1]]:
                for s3 in domains[2] if query_vector[2] < 0 else [query_vector[2]]:
                    for s4 in domains[3] if query_vector[3] < 0 else [query_vector[3]]:
                        for cond in domains[4] if query_vector[4] < 0 else [query_vector[4]]:
                            assignment = [s1, s2, s3, s4, cond]
                            if assignment[query_index] == qval:
                                total += joint_prob[s1, s2, s3, s4, cond]
        result[qval] = total

    total_prob = sum(result.values())
    if total_prob > 0:
        for k in result:
            result[k] /= total_prob

    return result

# EXAMPLE USAGE
query_vector = [1, 1, -1, 1, -2]  # Fever=1, Cough=1, Fatigue=?, Breathing=1, query Disease
posterior = infer_from_joint(joint_prob, query_vector, num_diseases)

# Show top 5 diseases with probabilities
top_results = sorted(posterior.items(), key=lambda x: -x[1])[:5]
for i, prob in top_results:
    print(f"{le_disease.inverse_transform([i])[0]}: {prob:.4f}")
