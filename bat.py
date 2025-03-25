import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (update the filename if necessary)
df = pd.read_csv("batter_player_stats.csv")

# Extract necessary columns
x = df["matches_lost"]
y = df["strike_rate"]

# Compute linear regression using formulas
n = len(x)
mean_x, mean_y = np.mean(x), np.mean(y)

# Calculate slope (m) and intercept (b)
m = (np.sum(x * y) - n * mean_x * mean_y) / (np.sum(x**2) - n * mean_x**2)
b = mean_y - m * mean_x

# Generate regression line
y_pred = m * x + b

# Plot line plot using seaborn
plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=y, label='Actual Data', color='blue')
sns.lineplot(x=x, y=y_pred, label=f'Regression Line (y = {m:.2f}x + {b:.2f})', color='red')
plt.xlabel("Matches Lost")
plt.ylabel("Strike Rate")
plt.title("Matches Lost vs Strike Rate with Linear Regression")
plt.legend()
plt.grid()
plt.show()
