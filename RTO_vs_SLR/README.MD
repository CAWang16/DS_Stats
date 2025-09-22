# Regression Through Origin vs. Simple Linear Regression

## Project Overview
This project explores when to use regression through the origin (RTO) versus simple linear regression (SLR). Using simulated data from the model Y = βX + ϵ, I applied hypothesis testing to determine whether an intercept term is necessary.  

## Objectives
1. Simulate dataset from Y = βX + ϵ with:  
   - β = 2, σ² = 1, n = 100, X ∈ [0, 100].  
2. Fit both SLR (with intercept) and RTO (no intercept) models.  
3. Use statistical tests to decide model appropriateness:  
   - t-test on intercept (H₀: β₀ = 0).  
   - Partial F-test comparing SLR vs RTO.  
4. Construct 95% confidence interval for slope (β₁).  

## Methods
- Simulation with NumPy (random uniform X, Gaussian noise ε).  
- Model fitting via least squares.  
- Hypothesis testing with t-statistics and F-statistics.  
- Interval estimation using Student’s t-distribution.  

## Results
- If the intercept is not significantly different from 0 (p > 0.05), RTO is appropriate.  
- Otherwise, use full SLR with intercept.  
- Confidence intervals quantify the precision of slope estimates.  

## Real-World Implication
This analysis is critical in fields where theory suggests no intercept:  
- Physics: Voltage vs Current (Ohm’s law).  
- Economics: Cost directly proportional to quantity.  
- Chemistry: Reaction rate proportional to concentration.  

By applying hypothesis tests, analysts can statistically justify model selection, avoiding bias from forcing or excluding intercept terms.  

## Files
- `hw2-q6.py` → Python implementation of simulation, regression, and tests.  
