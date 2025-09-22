# Import library
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
## Read the csv
data = pd.read_csv("problem4-oxygenpurity.csv")
## Define the notations
df = pd.DataFrame(data)
x = df['hydro']
y = df['purity']
x_bar = np.mean(x)
y_bar = np.mean(y)
Sxy = np.sum((x-x_bar)*(y-y_bar))
Sxx = np.sum((x-x_bar) ** 2)
b1_hat = Sxy / Sxx
b0_hat = y_bar - b1_hat*x_bar
y_hat = b0_hat + b1_hat*x
n = len(x)

# ANOVA
sse = np.sum((y-y_hat) ** 2)
sst = np.sum((y-y_bar) ** 2)
ssr = sst - sse

# Error variance estimate
s2 = sse / (n - 2)

# Variance and SE coefficient
var_b1_hat = s2 / Sxx
se_b1_hat = var_b1_hat ** 0.5

var_b0_hat = s2 * (1/n + (x_bar**2) / Sxx)
se_b0_hat = var_b0_hat ** 0.5


r2 = ssr / sst


# Setting the x0 value
x0 = np.array([0.8, 0.85, 0.9, 0.95, 1])
y0_hat = b0_hat + b1_hat * x0

alpha = 0.05
t_crit = st.t.ppf( 1-alpha/2 , df = n -2)
s = s2 ** 0.5

# CI for E[y0]
se_mean = s * ((1/n + (x0-x_bar)**2/Sxx) ** 0.5)
se_pred = s * ((1 + (1/n) + (x0-x_bar)**2/Sxx) ** 0.5)

# bands
ci_lower = y0_hat - t_crit * se_mean
ci_upper = y0_hat + t_crit * se_mean
pi_lower = y0_hat - t_crit * se_pred
pi_upper = y0_hat + t_crit * se_pred

ci_width = ci_upper - ci_lower
pi_width = pi_upper - pi_lower
best_ci_x0 = x0[np.argmin(ci_width)]
best_pi_x0 = x0[np.argmin(pi_width)]
# Tabulate results
results = pd.DataFrame({
    "x0": x0,
    "y0_hat": y0_hat,
    "CI_low": ci_lower, "CI_up": ci_upper,
    "PI_low": pi_lower, "PI_up": pi_upper
})
results["CI_width"] = results["CI_up"] - results["CI_low"]
results["PI_width"] = results["PI_up"] - results["PI_low"]

# Which x0 is most precise (narrowest)?
best_ci_x0 = results.loc[results["CI_width"].idxmin(), "x0"]
best_pi_x0 = results.loc[results["PI_width"].idxmin(), "x0"]

print(results.round(4))
print(f"\nSample mean x̄ = {x_bar:.2f}")
print(f"Narrowest CI at x0 = {best_ci_x0}")
print(f"Narrowest PI at x0 = {best_pi_x0}")
plt.scatter(x, y, label="data", s=20)
plt.plot(x0, y0_hat, label="fit", linewidth=2)

plt.fill_between(x0, ci_lower, ci_upper, alpha=0.25, label="95% CI")
plt.fill_between(x0, pi_lower, pi_upper, alpha=0.15, label="95% PI")

plt.xlabel("Hydro")
plt.ylabel("Purity")
plt.legend()
plt.show()

closest_x0 = x0[np.argmin(np.abs(x0 - x_bar))]
print(f"Most precise at x0 ={closest_x0}, because it is the closest one to the mean." )
