import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler #, ClusterCentroids, TomekLinks
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

def cat_cont_cols(df):
    """
        This function return a 2 dim list containing
        categorial columns and continuous columns
    """
    cat_names = []
    cont_names = []
    for name in df.columns:
        if df[name].dtypes == 'O':
            cat_names.append(name)
        else:
            cont_names.append(name)
    return [cat_names, cont_names]


def del_observation(df, col, val): 
    """
        This fuction delete all the observation in the dataframe df
        with value val the the column col
        It returns the resultin dataframe
    """
    indexes = df[df[col] == val].index
    return df.drop(indexes)


def split_test_train(df, criterion, test_frac, random_state = 42):
    """
        Here we build a test set with test_frac% observations from df
        only based on the criterion col (categorial)
        return train and test sets
    """
    cats = df[criterion].unique()
    test= pd.DataFrame()
    for cat in cats:
        temp = df[df[criterion] == cat].sample(frac = test_frac, random_state = random_state)
        test = pd.concat([temp, test])
        df = df.drop(temp.index)
    return df, test


def knn_imputation(df, n_neighbors = 5, label_col= None, test = False):
    """
        Here we impute the missing values in df using knnImputer
        with uniform weights
        if the data set is too big, a plit into samples of size 10000
        the function should be called first on training set before setting test = True
    """
    data = df.sample(frac= 1, random_state=42)
    if label_col:
        labels = data[label_col].values
        data = data.drop(label_col, axis = 1)
    cols = data.columns
    n = df.shape[0] // 10000
    if not test:
        imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    x = []
    for i in range(n):
        data_temp = data[i*10000: (i+1)*10000]
        x.extend(imp.fit_transform(data_temp))
    data_temp = data[n*10000:]
    x.extend(imp.fit_transform(data_temp))
    data = pd.DataFrame(data = x, columns = cols)
    if label_col:
        data[label_col] = labels
    return data, imp


def resample(df, target, method = "over", random_state = 42):
    """
        This function performs under or over sampling
        df is the dataframe
        method = under or over
    """
    assert method == "over" or method == "under", "Wrong the sampling method (over / under)"
    np.random.seed(random_state)
    y_train = df[target].values
    X_train = df.drop(target, axis = 1).values
    if method == "under":
        sampler = RandomUnderSampler(random_state = random_state)
    elif method == "over":
        sampler = SMOTE(random_state = random_state)
    X_res, y_res = sampler.fit_sample(X_train, y_train)
    #print('Resampled dataset shape %s' % Counter(y_cc))
    return X_res, y_res


def cross_validation(model, X_train, y_train, cv, scoring='accuracy'):
    scores = cross_val_score(
    model, X_train, y_train, cv=cv, scoring=scoring,n_jobs=-1)
    return "%s : %0.2f (+/- %0.2f)" % (scoring, scores.mean(), scores.std() * 2)
    

def feature_importance(model, x_col):
    X_columns = x_col
    plt.figure(figsize=(15, 5))
    ordering = np.argsort(model.feature_importances_)[::-1][:50]
    importances = model.feature_importances_[ordering]
    feature_names = X_columns[ordering]
    x = np.arange(len(feature_names))
    plt.bar(x, importances)
    plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15)


def grid_search_wrapper(model,X_train, y_train, X_test, y_test, params, scorers, refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(model, params, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of the model optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


def preprocess(df, target, n_test, random_state, sample_train = 'under', del_col= None, del_obs = None):
    """
        This function performs data reprocessing 
        ie dummy encoding and knn imputation
        and return train and test set
    """
    if del_col:
        df_enc = df.drop(del_col, axis=1)
    #we move the target column to the last column
    temp = df_enc[target]
    df_enc = df_enc.drop(target, axis =1)
    df_enc[target] =temp
    # dummy encoding
    df_enc= pd.get_dummies(df_enc, drop_first=True)
    df_enc = df_enc.rename(columns={df_enc.columns[-1]: target})
    #spliting the df data set into train and test 
    train, test = split_test_train(df_enc, n_test,target, random_state)
    # imputing training set
    train, imp = knn_imputation(train, n_neighbors=5, label_col=target)
    # imputing test set
    cols = train.columns[:-1]
    y_test = test[target].values
    X_test = test.drop(target, axis = 1)
    X_test = imp.fit_transform(X_test)
    test= pd.DataFrame(data = X_test,columns=cols)
    test[target] = y_test
    #resampling train set
    X_train, y_train = resample(train,target, method=sample_train, random_state=random_state)
    train= pd.DataFrame(data = X_train,columns=cols)
    train[target] = y_train
    return train, test


def test_model(model, X_test, y_test):
    """
        This return the precision, the recall, the f1 score and the confusion matrix
         of of the model tested on X_test an y_test
    """
    y_pred= model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    conf = confusion_matrix(y_test, y_pred)
    return acc, prec,rec , f1, conf

def fit_test_models(models, X_train, y_train,  X_test, y_test):
    """
        Return the models fitted and tested with test_model
    """
    temp = {}
    X= []
    for model in models:
        m = models[model]
        m.fit(X_train, y_train)
        acc, prec, rec, f1 , mat = test_model(m, X_test, y_test)
        X.append([acc, prec, rec, f1, mat])
        temp[model] = m
    res = pd.DataFrame(data = X, columns=["accuracy","precision", "recall", "f1", "conf_mat"], index=models.keys() )
    models = temp
    return models, res

def get_models():
    """
        return a dictionary of models
    """
    Logistic_reg = LogisticRegression()
    dtree = DecisionTreeClassifier(max_depth=3)
    lgbm = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=5, 
                                learning_rate=0.01, n_estimators=200,
                                max_bin = 100, bagging_fraction = 0.9,
                                bagging_freq = 1, feature_fraction = 0.6,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf =3)
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, max_features=50, n_jobs=-1)
    models = {"logistic_regresseion":Logistic_reg,
            "decision_tree":dtree,
            "light_gradient_boosting":lgbm,
            "Gradient_boosting":gb,
            "random_forest": rf}
    return models