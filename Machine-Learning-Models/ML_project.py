# -*- coding: utf-8 -*-
"""
@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

# Read Excel file #####
df = pd.read_excel('dataset/Data_Train.xlsx')

''' data summarization '''
def summary(df):
    print('--- Description of numerical variables')
    print(df.describe())
    print('--- Description of categorical variables')
    print(df.describe(include=['object']))
    print('--- Gerenal information about variables')
    print(df.info())
    print('--- view the 5 rows of dataset')
    print(df.head(5))
    return None

''' data cleaning '''
def datacleaning(df, special_character1=False, digit=False,
                 nalpha=False):
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True)
    df_cleaned = df.copy()
    if special_character1:
        df_cleaned = df_cleaned.apply(lambda x: x.str.split(r'[^a-zA-Z.\d\s]').
                                      str[0] if(x.dtypes == object) else x)
    if digit:
        df_cleaned = df_cleaned.apply(lambda x: x.str.replace(r'\d+', '')
                                      if(x.dtypes == object) else x)
    if nalpha:
        df_cleaned = df_cleaned.apply(lambda x: x.str.replace(r'\W+', ' ')
                                      if(x.dtypes == object) else x)
    df_cleaned = df_cleaned.apply(lambda x: x.str.strip() if(x.dtypes == object)
                                  else x)
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('New Delhi', 'Delhi')
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('Delhi NCR', 'Delhi')
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('Kochi', 'Cochin')
    return df_cleaned

''' missing values treatment '''
def treat_missingValue(df):
    # for numerical columns
    df = df.apply(lambda x: x.fillna(x.median())
				  if(x.dtypes != object) else x)
    # for categorical columns
    df = df.dropna()
    return df


''' outlier detection & treatment '''
def detect_outliers(col):
    outlier1 = []
    threshold = 3
    mean = np.mean(col)
    std =np.std(col)
    for i in col:
        z_score = (i - mean)/std 
        if np.abs(z_score) > threshold:
            outlier1.append(i)
    outlier2 = []
    sorted(col)
    q1, q3 = np.percentile(col,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr)
    for i in col:
        if ((i > upper_bound) | (i < lower_bound)):
            outlier2.append(i)
    lst1 = np.unique(outlier1)
    lst2 = np.unique(outlier2)
    output = list(set(lst1) & set(lst2))
    return output 

''' treat outliers '''
def treat_outliers(df):
	df = df.apply(lambda x: x.replace(detect_outliers(x), x.median())
				  if(x.dtypes != object) else x)
	return df

''' Get categorical columns '''
def get_categoricals(df):    
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
             categoricals.append(col)
        else:
             df[col].fillna(0, inplace=True)
    return categoricals


''' feature engineering '''
def featureEngineering(df):
    vote = df.VOTES.str.split(' ', n = 1, expand = True)
    df['VOTES'] = vote[0]
    df['VOTES'] = df['VOTES'].astype('int64')
    df['RATING'] = df['RATING'].astype('float64')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['months'] = df['Timestamp'].dt.month_name()
    df['day'] = df['Timestamp'].dt.day_name()
    df = df.sort_values(by='Timestamp')
    # Label Encoding
    lb_make = LabelEncoder()
    df.CITY = lb_make.fit_transform(df.CITY)
    # One hot Encoding
    df = pd.get_dummies(df, columns=get_categoricals(df), dummy_na=True)
    return df

''' Feature Selection '''
def Feature_selection(df):
    #Backward Elimination
    X = df.iloc[:,1:11]
    y = df.iloc[:,11]
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    print('selected_features_set1:' + str(cols))
    
    # using statistics method
    X_new = SelectKBest(chi2, k=6).fit_transform(X, y)
    cols = X_new.columns
    print('selected_features_set2:' + str(cols))
    
    # Recursive Feature Elimination
    model = LogisticRegression()
    rfe = RFE(model, 6)
    fit = rfe.fit(X, y)
    print("Feature Ranking: %s" % (fit.ranking_))
    return None

summary(df)
cleaned_df = datacleaning(df)
cleaned_df = treat_missingValue(cleaned_df)
cleaned_df = treat_outliers(cleaned_df)
feature = featureEngineering(cleaned_df)
selected_features = Feature_selection(feature)

''' Exploratory Data Analysis '''
f, axes = plt.subplots(2, 2)
sns.boxplot(x=feature['COST'], ax=axes[0, 0])
sns.boxplot(x=feature['VOTES'], ax=axes[0,1])
sns.boxplot(x=feature['RATING'] , ax=axes[1,0])
sns.pairplot(feature,vars=['COST','VOTES','RATING'], kind='scatter')
sns.countplot(y=feature["CITY"])
sns.countplot(y=feature["CUISINES"])
sns.lineplot(x='months',y='COST',data=feature, estimator=np.median)
sns.barplot(x="COST", y="CITY", data=feature, estimator=np.median)
table=pd.crosstab(feature["CUISINES"], feature['CITY'])
table.plot(kind='barh',stacked=True)

''' Split the training and tesing datasets from main datasets '''
X,y = selected_features[cols], selected_features['COST']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)


## import support vector machine #####
model = SVC()

'''
there are many optimization methods such as: 
1. Grid search 2. random search 3. Bayesian optimization
'''
def Gridsearch_SVM(model):
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
## gridsearchCV & cross_validate
    grid = GridSearchCV(model, parameters, cv=5)
    grid_result = grid.fit(X_train, y_train)
# summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid

# run gridsearch function to find the optimum value of model parameters
grid = Gridsearch_SVM(model)
predictions = grid.predict(X_test)

# check the accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## model evaluation
cm = confusion_matrix(y_test, predictions)
Accuracy_Score = accuracy_score(y_test, predictions)
Recall = recall_score(y_test, predictions, average='weighted')
Precision = precision_score(y_test, predictions, average='weighted')

# Build boosting Algorithms
ada_boost = AdaBoostRegressor() 
grad_boost = GradientBoostingRegressor()
xgb_boost = XGBRegressor(random_state=1,learning_rate=0.01)
boost_array = [ada_boost, grad_boost, xgb_boost]

# Ensemble the 3 boosting algos to optimize the accuracy of model
EnsembleMethod = VotingRegressor(estimators=[('Ada Boost', ada_boost),
                                   ('Grad Boost', grad_boost), ('XG Boost', xgb_boost)])

# Cross validation to find best score of all models
labels = ['Ada Boost', 'Grad Boost', 'XG Boost', 'Ensemble']
for clf, label in zip([ada_boost, grad_boost, xgb_boost, EnsembleMethod], labels):
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(),
          scores.std(), label))

# Train Ensemble model to predict the output
Final_model = EnsembleMethod.fit(X_train, y_train)
predictions = Final_model.predict(X_test)

## Ensemble model evaluation
cm = confusion_matrix(y_test, predictions)
Accuracy_Score = accuracy_score(y_test, predictions)
Recall = recall_score(y_test, predictions, average='weighted')
Precision = precision_score(y_test, predictions, average='weighted')

# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except Exception as e:

            return jsonify({'trace': str(e)})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)