"""
Jesus Ortega - 80421772
Agustin Omar Marquez - 80575895
Orion Wood - 80537518

CS4320 - Final Project - Health Diagnostic Decision Support

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
    for queryValue in domains[queryIndex]:
        total = 0.0
        for s1 in domains[0] if queryVector[0] < 0 else [queryVector[0]]:
            for s2 in domains[1] if queryVector[1] < 0 else [queryVector[1]]:
                for s3 in domains[2] if queryVector[2] < 0 else [queryVector[2]]:
                    for s4 in domains[3] if queryVector[3] < 0 else [queryVector[3]]:
                        for cond in domains[4] if queryVector[4] < 0 else [queryVector[4]]:
                            assignment = [s1, s2, s3, s4, cond]
                            if assignment[queryIndex] == queryValue:  #if the assignment matches the query value, add probability to total
                                total += jointProbability[s1, s2, s3, s4, cond]
        result[queryValue] = total    #result for this query value equals the total probability

    totalProbability = sum(result.values()) #sum up all (unnormalized) probabilities
    if totalProbability > 0:    #if the total probability is greater than 0, normalize the result
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
conditionResult = enumerationInference(jointProbability, qv, numDiseases)
sortedConditions = sorted(conditionResult.items(), key=lambda x: -x[1])  # Sort by descending probability
for k, v in sortedConditions[:5]:
    diseaseName = diseaseLabeler.inverse_transform([k])[0]
    print(f"  {diseaseName}: {v * 100:.2f}%")

"""
# EXAMPLE USAGE
queryVector = [1, 1, -1, 1, -2]  # Fever=1, Cough=1, Fatigue=?, Breathing=1, query Disease
posterior = enumerationInference(jointProbability, queryVector, numDiseases)

# Show top 5 diseases with probabilities
topResults = sorted(posterior.items(), key=lambda x: -x[1])[:5]
for i, prob in topResults:
    print(f"{diseaseLabeler.inverse_transform([i])[0]}: {prob*100:.2f}%")
"""

#User input for symptoms
def userDiagnoser(jointProbability, diseaseLabeler, numDiseases):
    #prompts the user for their symptoms
    print("\n --- Diagnistic Agent ---\n Please answer the following questions with 'yes', 'no', or 'unknown':")
    symptomPrompts = [
        "Do you have a fever?\n",
        "Do you have a cough?\n",
        "Do you feel fatigued?\n",
        "Do you have difficulty breathing?\n"
    ]
    userSymptoms = []
    for prompt in symptomPrompts:
        while True:
            response = input(prompt).strip().lower()
            if response == 'unknown' or response == 'u':
                userSymptoms.append(-1)
                break
            elif response == 'no' or response == 'n':
                userSymptoms.append(0)
                break
            elif response == 'yes' or response == 'y':
                userSymptoms.append(1)
                break
            else:
                print("Invalid input. Please answer with 'yes', 'no', or 'unknown'.")
    userSymptoms.append(-2)  # Placeholder for disease query
    # Perform inference using user symptoms
    userPosterior = enumerationInference(jointProbability, userSymptoms, numDiseases)
    # Show top 5 diseases with probabilities
    print("\nTop 5 Most Likely Conditions Based on Your Symptoms, and Associated Percentages: ")
    topDiseases = sorted(userPosterior.items(), key=lambda x: -x[1])[:5]
    for i, probability in topDiseases:
        diseaseName = diseaseLabeler.inverse_transform([i])[0]
        print(f"  {diseaseName}: {probability * 100:.2f}%")
#loop to allow multiple diagnoses, if wanted
while True:
    userDiagnoser(jointProbability, diseaseLabeler, numDiseases)
    goAgain = input("\nWould you like to diagnose someone else? (yes/no): ").strip().lower()
    if goAgain != 'yes' and goAgain != 'y':
        print("Ending diagnostic agent. Goodbye!")
        break