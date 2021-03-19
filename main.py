import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

data = pd.read_csv('data-sets/heart_failure.csv')

# Check for missing values
missing_values_count = data.isnull().sum()

# Check for duplicate values
dups = data.duplicated().sum()

print('Missing values: \n', missing_values_count, 'Duplicates: ', dups)

# Checking the correlation and skewness of data
corr = data.corr()
skew = data.skew()

print('Correlation: \n', corr)
print('Skewness: \n', skew)

# There is a visible disproportion with the DEATH_EVENT as well as some other values
# Following steps are: 1. plotting the data to see the distributions, 2. trying to minimize differences

def feature_plt(feature):  # code to visualize distribution, scatterplot and boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=110)

    sns.distplot(data[feature], ax=ax1)
    sns.boxplot(data[feature], orient='h', ax=ax2, width=0.2)

    print('Feature: ', feature,
          '\nSkewness: ', data[feature].skew())

    ax1.set_yticks([])
    return plt

# Example of how the plots look like
feature_plt('serum_creatinine')
plt.show()

# I went trough the ones with the biggest distance from 0, knowing their distribution lets me remove the outliers
data = data[data['creatinine_phosphokinase'] < 2000]
data = data[(data['platelets'] > 100000) & (data ['platelets'] < 450000)]
data = data[data['ejection_fraction'] < 65]
data = data[data['serum_sodium'] > 126]
data = data[data['serum_creatinine'] < 2]

print('Skewness after changes:\n', data.skew())

# Split data into variables
X = data.drop(['DEATH_EVENT'], axis=1)
y = data['DEATH_EVENT']

# The distribution between patients who died and those who didn't is still big
# I'm redistributing the data so the two are closer to each other, in this case I will oversample by creating new data
smote = SMOTE()

# After fitting both variables we can see that the numbers of 1 and 0 became equal
X_smote, y_smote = smote.fit_resample(X, y)
print('Data after reshaping:', Counter(y_smote))

steps = [('scaler', StandardScaler()),
         ('rfc', RandomForestClassifier())]
pipeline = Pipeline(steps)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=0, stratify=y_smote)
pipeline.fit(X_train, y_train)

# k-Cross validation
accuracies = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=10)
print('Accuracy:', round(accuracies.mean()*100, 2))
print('Standard deviation:', round(accuracies.std()*100, 2))

# Grid search to find the best params
params = [
    {'rfc__n_estimators': [100, 200, 300, 700],
     'rfc__max_features': ['auto', 'sqrt', 'log2']}]

grid = GridSearchCV(pipeline, param_grid=params, cv=10, n_jobs=-1)

# Train the model
grid.fit(X_train, y_train)

best_acc = grid.best_score_
best_params = grid.best_params_

print("Best accuracy: ", best_acc, "\nBest parameters: ", best_params)

# Score the test datas
test_data_result = grid.score(X_test, y_test)
print('Test data result: ', test_data_result)

# Scores around ~94%