#Jesus Ortega
#Omar Marquez
#Orion Wood
"""
This program loads a real-world dataset of patient symptom reports and diagnoses,
builds a joint probability distribution, and performs inference by enumeration
to determine the most likely diseases given observed symptoms.
"""
import pandas as pd
import numpy as np
from collections import defaultdict     # Automatically initializes new keys with a default value
from sklearn.preprocessing import LabelEncoder  # For converting string disease names to integers (or vice versa)

# Load dataset
data = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Select and convert symptom columns
symptomColumns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
dataSymptoms = data[symptomColumns + ["Disease"]].copy()
dataSymptoms[symptomColumns] = dataSymptoms[symptomColumns].applymap(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Encode disease labels
diseaseLabeler = LabelEncoder()
dataSymptoms["Disease"] = diseaseLabeler.fit_transform(dataSymptoms["Disease"])
uniqueDiseases = list(diseaseLabeler.classes_)
numDiseases = len(uniqueDiseases)

# Create joint probability table
jointProbability = np.zeros((2, 2, 2, 2, numDiseases))
counts = defaultdict(int)
for row in dataSymptoms.values:
    key = tuple(row)
    counts[key] += 1

total = len(dataSymptoms)
for key, count in counts.items():
    s1, s2, s3, s4, diag = key
    jointProbability[int(s1), int(s2), int(s3), int(s4), int(diag)] = count / total

# Inference function
def enumerationInference(jointProbability, queryVector, numDiseases):
    """
    Performs inference by enumeration over the joint probability distribution.
    Accepts avector where:
    -2 = query variable
    -1 = hidden variable (to be summed out)
     0/1/2/3 = evidence value for that variable
    Returns a dictionary mappingevery possible value of the query variable to its probability.
    """
    assert queryVector.count(-2) == 1, "Exactly one query variable must be specified"
    queryIndex = queryVector.index(-2)

    result = {}
    domains = {
        0: [0, 1],
        1: [0, 1],
        2: [0, 1],
        3: [0, 1],
        4: list(range(numDiseases)),
    }
    #for each possible value of the query variable, sum over all other variables (s1, s2, etc.)
    for qval in domains[queryIndex]:
        total = 0.0
        for s1 in domains[0] if queryVector[0] < 0 else [queryVector[0]]:
            for s2 in domains[1] if queryVector[1] < 0 else [queryVector[1]]:
                for s3 in domains[2] if queryVector[2] < 0 else [queryVector[2]]:
                    for s4 in domains[3] if queryVector[3] < 0 else [queryVector[3]]:
                        for cond in domains[4] if queryVector[4] < 0 else [queryVector[4]]:
                            assignment = [s1, s2, s3, s4, cond]
                            if assignment[queryIndex] == qval:  #if the assignment matches the query value, add probability to total
                                total += jointProbability[s1, s2, s3, s4, cond]
        result[qval] = total    #result for this query value equals the total probability

    totalProbability = sum(result.values())
    if totalProbability > 0:
        for k in result:
            result[k] /= totalProbability

    return result


print("\nMarginal Probabilities for Symptoms:")
for i in range(4):
    qv = [-1, -1, -1, -1, -1]
    qv[i] = -2  # Querying this symptom
    result = enumerationInference(jointProbability, qv, numDiseases)
    print(f"Symptom {i+1}:")
    for k, v in result.items():
        print(f"  Value {k}: {v * 100:.2f}%")
print("\nTop 5 Most Likely Conditions (Marginal Probabilities):")
qv = [-1, -1, -1, -1, -2]  # Querying the disease
condition_result = enumerationInference(jointProbability, qv, numDiseases)
sorted_conditions = sorted(condition_result.items(), key=lambda x: -x[1])  # Sort by descending probability
for k, v in sorted_conditions[:5]:
    disease_name = diseaseLabeler.inverse_transform([k])[0]
    print(f"  {disease_name}: {v * 100:.2f}%")




# EXAMPLE USAGE
query_vector = [1, 1, -1, 1, -2]  # Fever=1, Cough=1, Fatigue=?, Breathing=1, query Disease
posterior = enumerationInference(jointProbability, query_vector, numDiseases)

# Show top 5 diseases with probabilities
top_results = sorted(posterior.items(), key=lambda x: -x[1])[:5]
for i, prob in top_results:
    print(f"{diseaseLabeler.inverse_transform([i])[0]}: {prob*100:.2f}%")
