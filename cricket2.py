import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\Amrit Singhal\Desktop\Machine Learning\each_ball_records.csv")  # Ensure this is the correct CSV file

# Check column names
df.columns = df.columns.str.strip()  # Remove extra spaces if any

# Line plot: Ball Number vs. Score
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["ballnumber"], y=df["score"], marker="o", color="blue")

# Customize plot
plt.xlabel("Ball Number")
plt.ylabel("Score")
plt.title("Score Progression Over Balls in IPL 2023")
plt.grid(True)
plt.show()
