import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("Zomato-data-.csv")

# Convert 'rate' column to numeric (handling invalid values like 'NEW', '-')
df["rate"] = pd.to_numeric(df["rate"], errors='coerce')

# Drop rows where 'rate' or 'votes' is NaN or zero
df = df.dropna(subset=["rate", "votes"])# Remove restaurants with zero votes

# Check if data exists after cleaning
if df.empty:
    print("No valid data found after cleaning. Please check the dataset.")
else:
    # Extract features
    X = df["rate"]
    y = df["votes"]

    # Compute linear regression manually
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X * y)
    sum_x2 = np.sum(X ** 2)

    # Calculate slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x ** 2))
    b = (sum_y - m * sum_x) / n

    # Predicted values
    y_pred = m * X + b

    # Plot using Lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=X, y=y, marker="o", color="blue", label="Actual Votes")
    sns.lineplot(x=X, y=y_pred, color="red", label="Regression Line")

    # Customize plot
    plt.xlabel("Rate")
    plt.ylabel("Votes")
    plt.title("Linear Regression: Rate vs Votes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print regression equation
    print(f"Linear Regression Equation: Votes = {m:.4f} * Rate + {b:.4f}")
