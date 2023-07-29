# Overview 
Tools for implementing upper confidence bound optimization

# Contents
* UCB_opt_tools.py - Python code containing tools for performing UCB optimization
* Demos - datasets with examples of how UCB_opt_tools can be employed 
  * ATR_Engineering - Acyl-ACP reductase dataset from Greenhalgh et al. 2021 (Nat. Comm.) with simulation of UCB walk

# Software Requirements 
This code has been tested on the following system(s): 
 * MacOS Ventura (13.3.1)

## Python Dependencies 

This code has been tested using the specified version of the following packages: 
* numpy 1.20.1
* pandas 1.2.4
* matplotlib 3.5.1
* seaborn 0.11.1
* scikit-learn 0.24.1

# Quick start guide: 
* Place UCB_opt_tools.py in the same path as the file you are implementing it
* Perform any preprocessing (scaling, encoding). You'll need a feature array (X_trn) and a labels vector (y_trn) for training and a feature array for making predictionis (X_pred)
* Initialize the model by running: 
```
ucb = GetUCB() # Initialize 
ucb.fit(X_trn, y_trn).transform(X_pred) # Fit and make predictions 
```



* 

  



