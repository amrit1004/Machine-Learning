import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV file
df = pd.read_csv("h.csv")  # Replace 'your_file.csv' with the actual file path

# Extracting feature1 (Country) and feature2 (Happiness Score)
df["Country_Abbrev"] = df["country"].str[:3]  # Use only the first three letters of the country name
x = df["Country_Abbrev"]
y = df["HappiestCountriesWorldHappinessReportScore2024"]

# Plotting Line Graph
plt.figure(figsize=(10, 5))
plt.lineplot(x, y, color='blue', marker='o', linestyle='-', label='Country vs Happiness Score')
plt.xlabel("Country (Abbreviated)")
plt.ylabel("Happiness Score")
plt.title("Happiness Score by Country (2024)")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()