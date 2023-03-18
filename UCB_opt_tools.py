def UCB_opt(X_trn, y_trn, X_pred, num_std=1, **gpr_kwargs):
    ''' Function for identifying the optimal upper-confidence bound
        Inputs:
        X_trn - array of features for training the GP regressor. Row order must correlate with y_trn
        y_trn - series or vector containing the labels for the GP regressor. Row order must correlate with X_trn
        X_pred - array of features to be used for prediction
        num_std - number of standard deviations to use as a multiplier in upper-confidence bound estimation. Default: 1
        **gpr_kwargs - kwargs that feed into GuassianProcessRegressor (such as kernel, alpha, etc.)
        
        Outputs:
        X_pred_opt - feature encoding for the sample with the maximum UCB 
        y_pred_opt - the mean predicted value corresponding to the maximum UCB
        UCB_opt_ind - the row index in X_pred corresponding to the maximum UCB 
        '''

    # Intialize the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(**gpr_kwargs)

    # Train and predict
    y_pred, std_pred = gpr.fit(X_trn, y_trn).predict(X_pred, return_std=True)

    # Calculate UCB (defined as the sum of the mean and the confidence interval)
    UCB = y_pred + num_std*std_pred

    # Find the index of the UCB optimum
    UCB_opt_ind = UCB.argmax()

    # Find the mean of the prediction corresponding to the UCB optimum
    y_pred_opt = list(y_pred)[UCB_opt_ind]
    X_pred_opt = np.array(X_pred)[UCB_opt_ind]

    
    return X_pred_opt, y_pred_opt, UCB_opt_ind
  
  
def UCB_batch_mode(X_trn, y_trn, X_pred, batchsize, num_std=1, **gpr_kwargs):

    '''  Function for identifying the optimal upper-confidence bound
        Inputs:
        X_trn - array of features for training the GP regressor. Row order must correlate with y_trn
        y_trn - series or vector containing the labels for the GP regressor. Row order must correlate with X_trn
        X_pred - array of features to be used for prediction
        batchsize - number of samples to use in the batch
        num_std - number of standard deviations to use as a multiplier in upper-confidence bound estimation. Default: 1
        **gpr_kwargs - kwargs that feed into GuassianProcessRegressor (such as kernel, alpha, etc.)
        
        Outputs:
        UCB_opt_batch - list of indices that are batch-mode UCB optima for a given round
     
    '''
       
    # Initialize variables 
    batch = 1 # counter
    X_trn_batch = np.array(X_trn)   # training set features
    y_trn_batch = list(y_trn)       # training set labels 
    X_pred_batch = np.array(X_pred) # prediction set 

    UCB_opt_ind_batch = [] # batch-mode UCB optima
    
    while batch <= batchsize:
        # Optimize the UCB 
        X_top_pred, y_pred, UCB_opt_ind = UCB_opt(X_trn_batch, y_trn_batch, X_pred_batch, num_std=num_std, **gpr_kwargs)

    
        # Add a psuedo-measurement to the training data by assuming that the predicted value for the UCB optimum will become the new observed value
        # Update the labels with the pseudo-measurement
        y_trn_batch.append(y_pred)

        # Update the the feature list with the pseudo-measurement features
        X_trn_batch = np.vstack([X_trn_batch, X_top_pred])

        # Update the batch-mode list
        UCB_opt_ind_batch.append(UCB_opt_ind)
    

        # Update the counter
        batch += 1

    return X_trn_batch,y_trn_batch, UCB_opt_ind_batch
    




