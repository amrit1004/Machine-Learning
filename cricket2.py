import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\Amrit Singhal\Desktop\Machine Learning\each_ball_records.csv")

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Extract variables
X = df["ballnumber"]
y = df["score"]

# Compute necessary sums
n = len(X)
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x2 = np.sum(X ** 2)

# Calculate slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

# Predict values using the regression equation
y_pred = m * X + b

# Plot actual vs predicted using lineplot
plt.figure(figsize=(12, 6))
sns.lineplot(x=X, y=y, marker="o", color="blue", label="Actual Scores")  # Actual scores
sns.lineplot(x=X, y=y_pred, color="red", label="Regression Line")  # Regression line

# Customize plot
plt.xlabel("Ball Number")
plt.ylabel("Score")
plt.title("Linear Regression: Score Progression Over Balls in IPL 2023")
plt.legend()
plt.grid(True)
plt.show()

# Print equation
print(f"Linear Regression Equation: y = {m:.4f}x + {b:.4f}")
