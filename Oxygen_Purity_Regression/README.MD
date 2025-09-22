# Oxygen Purity Regression Analysis

## Project Overview
This project analyzes the relationship between hydrogen input and oxygen purity in a chemical process. Using simple linear regression (SLR), I modeled how changes in hydrogen concentration affect purity levels and visualized both confidence intervals (CI) and prediction intervals (PI) for future estimates.  

## Objectives
- Fit a simple linear regression model to the oxygen purity dataset.  
- Generate 95% confidence intervals for the mean response at target hydrogen levels.  
- Generate 95% prediction intervals for individual future purity outcomes.  
- Compare the precision of intervals at five candidate input points: {80%, 85%, 90%, 95%, 100%}.  

## Methods
- Regression coefficients estimated using least squares.  
- Variance estimates derived from ANOVA decomposition.  
- Confidence and prediction intervals calculated using the t-distribution.  
- Visualization of regression line with shaded CI and PI bands.  

## Results
- Predictions are most precise near the mean hydrogen input level.  
- Interval width increases at the extremes (80% and 100%), showing greater uncertainty.  
- CI bands represent uncertainty around the average purity, while PI bands represent uncertainty around individual outcomes.  

## Real-World Implication
In manufacturing and chemical processing, these results help identify operating ranges where predictions are most reliable. Plant managers can use interval analysis to ensure quality control and anticipate variability when operating away from average conditions.  

## Files
- `hw2-q1.py` → Python implementation of regression, CI, and PI.  
- `problem4-oxygenpurity.csv` → Input dataset.  
