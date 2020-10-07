from sklearn.ensemble import RandomForestClassifier

# Random Forest classifer model
def RandomForest(test_set, train_set, features, target, params={'max_depth':2}):
    # Set X and Y
    X = train_set[features]
    Y = train_set[target]

    # Set model parameters
    max_depth = params['max_depth']

    # Fit RF model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X,Y)

    # Make predictions
    Xtest = test_set[features]
    pred = clf.predict_proba(Xtest)
    test_set['prediction'] = pred[:,1]
    return(test_set, clf)