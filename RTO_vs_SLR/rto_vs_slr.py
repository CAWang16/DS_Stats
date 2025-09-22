import numpy as np
from scipy import stats

np.random.seed(42)
n = 100
b = 2
sigma = 1

x = np.random.uniform(0,100, size = n)
eps = np.random.normal(0, sigma, size=n)
y = b * x + eps

# SLR
y_bar = y.mean()
x_bar = x.mean()
sxx = np.sum((x - x_bar)**2)
sxy = np.sum((x-x_bar)*(y-y_bar))
b1 = sxy/sxx
b0 = y_bar - b1*x_bar
y_hat = b0 + b1 * x

# ANOVA
sse = np.sum((y - y_hat)**2)
ssr = np.sum((y_hat - y_bar)**2)
sst = sse + ssr
df = n - 2
s2 = sse / df
se_b1 = (s2 / np.sum((x-x_bar)**2)) ** 0.5
se_b0 = (s2 * ((1/n) + ((x_bar)**2 / sxx))) **0.5

# CI SLR
t_critical = stats.t.ppf(0.975, df)
ci_b1 = (b1 - t_critical * se_b1, b1 + t_critical * se_b1)
ci_b0 = (b0 - t_critical*se_b0, b0 + t_critical*se_b0)

# t-test on b0
t_b0 = (b0 - 0) / se_b0
p_b0_two_sided = 2 * stats.t.sf(abs(t_b0), df)

print(ci_b1)

# RTO

b1_rto = np.sum(x*y)  / np.sum(x**2)
y_hat_rto = b1_rto * x
sse_rto = np.sum((y-y_hat_rto)**2)

df_rto = n - 1
s2_rto = sse_rto / df_rto
se_b1_rto = (s2_rto / np.sum(x**2))**0.5
t_critical_rto = stats.t.ppf(0.975, df_rto)
ci_b1_rto = (b1_rto - t_critical_rto * se_b1_rto, b1_rto + t_critical_rto * se_b1_rto)

# q = 1 here
f = ((sse_rto-sse)/1)/(sse/df)
p_f = stats.f.sf(f, 1, df)


# H0: b0 = 0 
# H1: b0 != 0
print(f"SLR b0: {b0}, CI: {ci_b0}") # includes 0
print(f"SLR b1: {b1}, CI: {ci_b1}")


print(f"SLR t for b0 = {t_b0}, p = {p_b0_two_sided}")
print(f"Partial F = {f}, p = {p_f}")
if p_b0_two_sided > 0.05 and p_f > 0.05:
    print(f"Since p > 0.05, fail to reject h0, means data do not provide sufficient evidence that the intercept differs from 0. ")
    print(f"RTO would be the model we use, and the line is y_hat = {b1_rto} * x")
    print(f"95% CI for b1:{ci_b1_rto}")


