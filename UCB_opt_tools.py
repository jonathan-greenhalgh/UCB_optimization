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

    def batch_mode(self, X_pred_initial, batch_size):

        self.X_pred = X_pred_initial
        self.batch_size = batch_size


        # Initialize variables for the batch-mode calculation
        batch = 1 # counter
        X_trn_batch = np.array(self.X)    
        y_trn_batch = list(self.y)
        X_pred_batch = np.array(X_pred_initial)

        UCB_opt_ind_batch = []
        X_top_batch = [] 
        y_top_batch = []

        while batch <= batch_size:
            # Optimize the UCB 
            self.fit(X_trn_batch, y_trn_batch).transform(X_pred_batch)
            X_top_batch.append(self.x_pred_opt)
            y_top_batch.append(self.y_pred_opt) 


            # Add psuedo-measurement to the training data 
            y_trn_batch.append(self.y_pred_opt)

            # Update the feature list
            X_trn_batch = np.vstack([X_trn_batch, self.x_pred_opt])

            # Update the batch-mode list
            UCB_opt_ind_batch.append(self.opt_ind)

            # Update the counter
            batch += 1

        if len(UCB_opt_ind_batch) != len(set(UCB_opt_ind_batch)):
            print('Warning: local optimum in batch')

            # Combine X_top_batch and y_top_batch into a dataframe 
            df_Xy = pd.DataFrame(X_top_batch)
            df_Xy['y'] = y_top_batch

            # Remove duplicates 
            df_Xy = df_Xy.drop_duplicates(subset = df_Xy.columns != 'y')
            print(f'Removed {batch_size-len(df_Xy)} duplicates')

            # Reassign X_top_batch and y_top_batch 
            X_top_batch = df_Xy.loc[:, df_Xy.columns != 'y'].to_numpy()
            y_top_batch = list(df_Xy['y'])

            UCB_opt_ind_batch = list(set(UCB_opt_ind_batch))



        self.X_batch = X_top_batch
        self.y_batch = y_top_batch
        self.UCB_opt_inds_batch = UCB_opt_ind_batch

        return self

