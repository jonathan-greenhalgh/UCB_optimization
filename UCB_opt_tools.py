import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import check_is_fitted



class GetUCB(BaseEstimator, TransformerMixin):
    '''A scikit-learn style class for training GPR models to determine Upper Confidence Bounds (UCB)'''
    def __init__(self, num_stds = 1, **gpr_kwargs):
        ''' Inputs:
            num_std - number of standard deviations to use as a multiplier in upper-confidence bound estimation. Default: 1
            **gpr_kwargs - kwargs that feed into GuassianProcessRegressor (such as kernel, alpha, etc.)
            '''
        #Initialize GPR
        self.gpr = GaussianProcessRegressor(**gpr_kwargs)
        self.num_stds = num_stds

    
    def fit(self, X, y,  **gpr_kwargs):
        ''' Inputs:
            X - array of features for training the GP regressor. Row order must correlate with y_trn
#           y - series or vector containing the labels for the GP regressor. Row order must correlate with X_trn'''

        self.X = X
        self.y = y

        return self
    
    def transform(self, X_pred):
        '''Transformation function for calculating UCB
            
            Inputs:
            X_pred - array of features to be used for prediction'''
        
        self.X_pred = X_pred

        # Fit the regression model
        self.fit_ = self.gpr.fit(self.X, self.y)

        self.y_preds, self.std_preds = self.fit_.predict(X_pred, return_std=True)

        # Calculate the UCB 
        self.UCB = self.y_preds + self.num_stds*self.std_preds

        # Find the index corresponding to the max UCB
        self.opt_ind = self.UCB.argmax()
        self.y_pred_opt = list(self.y_preds)[self.opt_ind]
        self.x_pred_opt = np.array(self.X_pred)[self.opt_ind]

        return self.UCB














 
# def UCB_opt(X_trn, y_trn, X_pred, num_std=1, **gpr_kwargs):
#     ''' Function for identifying the optimal upper-confidence bound
#         Inputs:
#         X_trn - array of features for training the GP regressor. Row order must correlate with y_trn
#         y_trn - series or vector containing the labels for the GP regressor. Row order must correlate with X_trn
#         X_pred - array of features to be used for prediction
#         num_std - number of standard deviations to use as a multiplier in upper-confidence bound estimation. Default: 1
#         **gpr_kwargs - kwargs that feed into GuassianProcessRegressor (such as kernel, alpha, etc.)
        
#         Outputs:
#         X_pred_opt - feature encoding for the sample with the maximum UCB 
#         y_pred_opt - the mean predicted value corresponding to the maximum UCB
#         std_preds - confidence interval for the prediction corresponding to the maximum UCB
#         opt_ind - the row index in X_pred corresponding to the maximum UCB 
#         '''

#     # Intialize the Gaussian Process Regressor
#     gpr = GaussianProcessRegressor(**gpr_kwargs)

#     # Train and predict
#     y_preds, std_preds = gpr.fit(X_trn, y_trn).predict(X_pred, return_std=True)

#     # Calculate UCB (defined as the sum of the mean and the confidence interval)
#     UCB = y_preds + num_std*std_preds

#     # Find the index of the UCB optimum
#     opt_ind = UCB.argmax()

#     # Find the mean of the prediction corresponding to the UCB optimum
#     y_pred_opt = list(y_preds)[opt_ind]
#     X_pred_opt = np.array(X_pred)[opt_ind]

    
#     return X_pred_opt, y_pred_opt, y_preds, std_preds, opt_ind

  
# def UCB_batch_mode(X_trn, y_trn, X_pred, batchsize, num_std=1, **gpr_kwargs):

#     '''  Function for identifying the optimal upper-confidence bound
#         Inputs:
#         X_trn - array of features for training the GP regressor. Row order must correlate with y_trn
#         y_trn - series or vector containing the labels for the GP regressor. Row order must correlate with X_trn
#         X_pred - array of features to be used for prediction
#         batchsize - number of samples to use in the batch
#         num_std - number of standard deviations to use as a multiplier in upper-confidence bound estimation. Default: 1
#         **gpr_kwargs - kwargs that feed into GuassianProcessRegressor (such as kernel, alpha, etc.)
        
#         Outputs:
#         X_top_batch - feature encoding for the samples with the maximum UCB 
#         y_top_batch - the mean predicted values corresponding to the maximum UCB
#         X_preds - encodings for all samples in prediction space
#         y_preds - mean predictions 
#         std_preds - confidence intervals for the prediction corresponding to the maximum UCB
#         UCB_opt_ind_batch - list of indices that are batch-mode UCB optima for a given round
     
#     '''
       
#     # Initialize variables 
#     batch = 1 # counter
#     X_trn_batch = np.array(X_trn)   # training set features
#     y_trn_batch = list( np.log(y_trn))       # training set labels 
#     X_pred_batch = np.array(X_pred) # prediction set 

#     UCB_opt_ind_batch = [] # batch-mode UCB optima
#     X_top_batch = [] 
#     y_top_batch = [] 

#     while batch <= batchsize:
#         # Optimize the UCB 
#         X_top_pred, y_pred_opt, y_preds, std_preds, opt_ind = UCB_opt(X_trn_batch, y_trn_batch, X_pred_batch, num_std=num_std, **gpr_kwargs)

#         # Add the top prediction to the batch
#         X_top_batch.append(X_top_pred)
#         y_top_batch.append(y_pred_opt)


#         # Add a psuedo-measurement to the training data by assuming that the predicted value for the UCB optimum will become the new observed value
#         y_trn_batch.append(y_pred_opt)

#         # Update the the feature list with the pseudo-measurement features
#         X_trn_batch = np.vstack([X_trn_batch, X_top_pred])

#         # Update the batch-mode list
#         UCB_opt_ind_batch.append(opt_ind)
    
#         # Update the counter
#         batch += 1


#     if len(UCB_opt_ind_batch) != len(set(UCB_opt_ind_batch)):
#         print('Warning: local optimum in batch')

#         # Combine X_top_batch and y_top_batch into a dataframe 
#         df_Xy = pd.DataFrame(X_top_batch)
#         df_Xy['y'] = y_top_batch

#         # Remove duplicates 
#         df_Xy = df_Xy.drop_duplicates(subset = df_Xy.columns != 'y')
#         print(f'Removed {batchsize-len(df_Xy)} duplicates')

#         # Reassign X_top_batch and y_top_batch 
#         X_top_batch = df_Xy.loc[:, df_Xy.columns != 'y'].to_numpy()
#         y_top_batch = list(df_Xy['y'])

#         UCB_opt_ind_batch = list(set(UCB_opt_ind_batch))

#     X_preds = X_pred_batch

#     # return X_trn_batch, y_trn_batch, X_preds, y_preds, std_preds, UCB_opt_ind_batch
#     return X_top_batch, y_top_batch, X_preds, y_preds, std_preds, UCB_opt_ind_batch

    




