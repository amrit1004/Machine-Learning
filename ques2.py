import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data
drive_exp = np.array([5, 2, 12, 9, 15, 6, 25, 16])
premium = np.array([64, 87, 50, 71, 44, 56, 42, 60])

# Compute means
x_mean = np.mean(drive_exp)
y_mean = np.mean(premium)

# Compute SSxx, SSyy, and SSxy
SSxx = np.sum((drive_exp - x_mean) ** 2)
SSyy = np.sum((premium - y_mean) ** 2)
SSxy = np.sum((drive_exp - x_mean) * (premium - y_mean))

# Compute regression coefficients
b = SSxy / SSxx
a = y_mean - b * x_mean

# Plot scatter plot and regression line
plt.scatter(drive_exp, premium, color='blue', label='Data points')
plt.plot(drive_exp, a + b * drive_exp, color='red', label='Regression line')
plt.xlabel('Driving Experience (years)')
plt.ylabel('Monthly Auto Insurance Premium')
plt.legend()
plt.show()

# Compute correlation coefficient and r^2
r = SSxy / np.sqrt(SSxx * SSyy)
r_squared = r ** 2

# Predict premium for 10 years of experience
y_pred_10 = a + b * 10

# Compute standard deviation of errors
n = len(drive_exp)
y_pred = a + b * drive_exp
errors = premium - y_pred
std_error = np.sqrt(np.sum(errors ** 2) / (n - 2))

# Compute 90% confidence interval for B
SE_b = std_error / np.sqrt(SSxx)
t_critical = stats.t.ppf(0.95, df=n-2)
CI_lower = b - t_critical * SE_b
CI_upper = b + t_critical * SE_b

# Hypothesis test for B negative
t_stat = b / SE_b
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
is_significant = p_value < 0.05

# Test whether correlation is different from zero
t_rho = r * np.sqrt((n - 2) / (1 - r ** 2))
p_rho = 2 * (1 - stats.t.cdf(abs(t_rho), df=n-2))

# Print results
results = {
    "SSxx": SSxx,
    "SSyy": SSyy,
    "SSxy": SSxy,
    "Intercept (a)": a,
    "Slope (b)": b,
    "Correlation (r)": r,
    "R-squared": r_squared,
    "Prediction for 10 years": y_pred_10,
    "Standard Error": std_error,
    "90% CI for B": (CI_lower, CI_upper),
    "Hypothesis Test B Negative": is_significant,
    "P-value for B": p_value,
    "P-value for rho ≠ 0": p_rho
}

for key, value in results.items():
    print(f"{key}: {value}")