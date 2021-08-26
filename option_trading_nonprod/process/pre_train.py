from sklearn.model_selection import GroupShuffleSplit

def splitDataTrainTestValOot(dataset, target = 'reachedStrikePrice', date_col='exportedAt', oot_share=0.1, test_share=0.8, val_share=0.8):
	# Split in train, validation, test and out of time
	# Take most recent observations for out of time set (apprx last 5000 observations)

	exportDateOot = dataset.iloc[-int(oot_share * len(dataset))][date_col]
	df_oot = dataset[dataset[date_col] >= exportDateOot]
	df_rest = dataset.drop(df_oot.index, axis=0).reset_index(drop=True)

	# test to split keeping exportedAt column always in same group
	gss_test = GroupShuffleSplit(n_splits=1, train_size=test_share, random_state=42)
	gss_test.get_n_splits()

	# split off test set
	test_groupsplit = gss_test.split(df_rest, groups = df_rest[date_col])
	train_idx, test_idx = next(test_groupsplit)
	df_rest2 = df_rest.loc[train_idx]
	df_test = df_rest.loc[test_idx]

	# test to split keeping exportedAt column always in same group
	gss_val = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=42)
	gss_val.get_n_splits()

	# split off validation set
	df_rest2 = df_rest2.reset_index(drop=True)
	val_groupsplit = gss_val.split(df_rest2, groups = df_rest2[date_col])
	train_idx, val_idx = next(val_groupsplit)
	df_train = df_rest2.loc[train_idx]
	df_val = df_rest2.loc[val_idx]

	# clean unwanted columns for model training
	# Add weights column
	X_train = df_train.drop(columns=[target])
	y_train = df_train[target]

	X_val = df_val.drop(columns=[target])
	y_val = df_val[target]

	X_test = df_test.drop(columns=[target])
	y_test = df_test[target]

	X_oot = df_oot.drop(columns=[target])
	y_oot = df_oot[target]

	print("Train shape: {}\nValidation shape: {}\nTest shape: {}\nOut of time shape: {}".format(X_train.shape,X_val.shape,X_test.shape,X_oot.shape))
	return X_train, y_train, X_test, y_test, X_val, y_val, X_oot, y_oot