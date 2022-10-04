# Author: Roman Smith

from pybaseball import cache, statcast, playerid_lookup
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


def main():
    # GET DATA

    #Pull from statcast
    cache.enable()
    data = statcast(start_dt='2022-04-07', end_dt='2022-10-05')
    
    # Get pitcher id
    LAST_NAME = 'Wainwright'
    FIRST_NAME = 'Adam'
    pitcher_id = playerid_lookup(last=LAST_NAME, first=FIRST_NAME).loc[0, 'key_mlbam']

    # Filter data by pitcher id
    data = data[data['pitcher'] == pitcher_id]

    # DATA PRE-PROCESSING

    # Create lists for the target feature and predictor features
    target = ['pitch_type']
    features = ['stand', 'p_throws', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number']

    # Subset the dataframe into a target dataset and input dataset
    y = data[target]
    x = data[features]

    # Change baserunning data to be 1 if any base runner or 0 if no baserunner
    x['on_3b'] = x['on_3b'].apply(lambda x: 0 if pd.isna(x) else 1)
    x['on_2b'] = x['on_2b'].apply(lambda x: 0 if pd.isna(x) else 1)
    x['on_1b'] = x['on_1b'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Generate one-hot encoder for 'stand' and 'p_throws'
    transformer = make_column_transformer(
        (OneHotEncoder(), ['stand', 'p_throws']),
        remainder='passthrough'
    )

    # Fit transformer to data
    transformed = transformer.fit_transform(x)

    # Reconstruct the input dataset using the transformed data and column names
    x = pd.DataFrame(
        transformed, 
        columns=transformer.get_feature_names_out()
    )

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # CALCULATE BASELINE ACCURACY

    df = y.value_counts(normalize=True).to_frame('counts')
    highest_proportion = df['counts'].max()
    print('Accuracy (predict highest percentage): ', highest_proportion)

    # BUILD MODEL (DEFAULT)

    # Initialize a classifier with default values and fit to the training data (random state 0 used for reproducability)
    clf_def = RandomForestClassifier(random_state=0)
    clf_def.fit(x_train, y_train.values.ravel())
    
    # Predict the target classes of the test data
    pred_def = clf_def.predict(x_test)

    # Measure performance
    acc_def = accuracy_score(y_test, pred_def)
    print('Accuracy (default parameters): ', acc_def)
    
    # BUILD MODEL (OPTIMIZED HYPERPARAMETERS)

    # Generate random grid (dict)
    random_grid = {
        'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)],
        'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'bootstrap': [True, False]
    }

    # Instantiate and fit randomized search cv
    clf_opt = RandomForestClassifier()
    clfs_opt = RandomizedSearchCV(estimator=clf_opt, param_distributions=random_grid, n_iter=200, cv=4, random_state=0, n_jobs=-1)
    clfs_opt.fit(x_train, y_train.values.ravel())

    # Find best estimator
    clf_opt = clfs_opt.best_estimator_

    # Predict the target classes of the test data
    pred_opt = clf_opt.predict(x_test)

    # Measure performance
    acc_opt = accuracy_score(y_test, pred_opt)
    print('Accuracy (optimized parameters): ', acc_opt)

if __name__ == '__main__':
     main()