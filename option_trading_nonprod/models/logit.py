import statsmodels.api as sm

# Logistic regressions model
def logitModel(test_set, train_set, features, target, params={}):
    X = train_set[features]
    Y = train_set[target]
    X = sm.add_constant(X)


    # Fit and summarize OLS model
    mod = sm.Logit(Y, X)

    res = mod.fit(maxiter=100)
    print(res.summary())

    # sometimes seem to need to add the constant
    Xtest = test_set[features]
    Xtest = sm.add_constant(Xtest, has_constant='add')
    pred = res.predict(Xtest)
    test_set['prediction'] = pred
    return(test_set, res)