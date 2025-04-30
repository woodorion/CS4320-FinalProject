import pandas as pd

#Load the dataset
csvFile = "Health_Data_set.csv" #Ensure the file is in the same directory as this script
data = pd.read_csv(csvFile)

#Display a preview
print("Preview of the dataset:")
print(data.head())

#Show dataset shape and columns
print("\nDataset info:")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")