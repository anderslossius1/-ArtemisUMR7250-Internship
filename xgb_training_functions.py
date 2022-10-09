"""
XGBOOST TRAINING-RELATED FUNCTIONS
CONTAINS:
    PHASE 1:
    - build_training_list() --> run_xgboost_training.py
    - split_data() --> run_xgboost_training.py
    - train() --> run_xgboost_training.py

@author: mgorlach & alossius
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def build_training_list(xgb_p, injected_waveforms):
    """PHASE 1
    Builds a list containing the names of all the waveforms that 
    will be used for model training.

    Parameters
    ----------
    xgb_p : dict
        The XGBoost params.
    injected_waveforms : list
        A list of the waveforms injected during this project's dag_inj.

    Returns
    -------
    train_list : list
        List of training waveforms.

    """
    print('\nBuilding training list...')

    # Initialize
    train_list = []

    # Append all waveforms that are in the training list (defined in xgb.yml) AND were actually injected (precautionary)
    for waveform, selection in xgb_p['TRAINING_LIST'].items():
        if (waveform in injected_waveforms) & (selection != None):
            train_list.append(waveform)
            
    print('Training list complete,', len(train_list), 'waveforms found for training.') 
    return train_list

def split_data(inj, bkg, train_list, p, debug_options):
    """PHASE 1
    Splits injection and background triggers into training and testing waveforms based on the desired training list
    
    Parameters
    ----------
    inj : Pandas DataFrame
        DataFrame containing all injection triggers
    bkg : Pandas DataFrame
        DataFrame containgin all background triggers
    train_list : list
        List containing all waveforms desired for training use
    p : float
        Percentage of total data that will be used for training
    debug_options : list
        List containing users preferences regarding debugging
        
    Returns
    -------
    df_train : Pandas DataFrame
        DataFrame containing all training triggers
    df_test : Pandas DataFrame
        DataFrame containg all testing triggers
    """
    print('\nCreating testing and training DataFrames...')
    
    # Read debug params
    debug, state = debug_options
    
    # Convert train_list to Pandas Series
    train_list = pd.Series(train_list)
    
    # Obtain sub-DataFrame containing all inj data for training waveforms only
    inj_train = inj[inj['kind'].isin(train_list)]
    
    # Run sklearn.model_selection train_test_split function (get a balanced 75% for training) 
    if debug: df_train = train_test_split(inj_train, stratify=inj_train['Signal'], random_state=state)[0]
    else:     df_train = train_test_split(inj_train, stratify=inj_train['Signal'])[0]
    
    # Drop all triggers that will be used for training - to obtain testing injection triggers
    inj_test = inj.drop(df_train.index)
    
    # Obtain certain fraction p of bkg data for training 
    if debug: bkg_train = bkg.sample(frac=p, random_state=state)
    else:     bkg_train = bkg.sample(frac=p)
    
    # Drop all triggers that will be used for training - to obtain testing background triggers
    bkg_test = bkg.drop(bkg_train.index)
    
    # Concat inj and bkg for both training and testing
    df_train = pd.concat([df_train,
                          bkg_train], axis=0)
    df_test = pd.concat([inj_test,
                         bkg_test], axis=0)
    
    print('Split complete')
    return df_train, df_test

def train(df_train, params):
    """PHASE 1
    Creates and trains an XGBoost Classifier from the given training data
    
    Parameters
    ----------
    df_train : Pandas DataFrame
        DataFrame containing all data (injections and background) for training model
    params : dict
        XGBClassifier params defined in xgb.yml
        
    Returns
    -------
    xgb_cl : XGBoost Classifier
        Trained XGBoost classifier than can be used to make predections
    """
    print('\nTraining XGBoost Model...')
    
    # Create independent and dependent DataFrames 
    x_train = df_train.drop(['Signal','alpha','kind'], axis=1)  # All data minus signal, alpha and kind columns
    y_train = df_train['Signal']                                # Only the signal column
    
    # Create a classifier, based on how the params are configured
    if params['REPRODUCEABLE']: xgb_cl = xgb.XGBClassifier(use_label_encoder=False, 
                                                           max_depth=params['MAX_DEPTH'], 
                                                           verbosity=params['VERBOSITY'], 
                                                           random_state=params['STATE_ID']) 
    else: xgb_cl = xgb.XGBClassifier(use_label_encoder=False, 
                                     max_depth=params['MAX_DEPTH'], 
                                     verbosity=params['VERBOSITY'])
    # Fit (train) the model
    xgb_cl.fit(x_train, y_train)
    
    print('Training complete')
    return xgb_cl