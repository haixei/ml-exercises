import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('data-sets/wine-quality.csv', sep=',')
data['quality'] = [1 if i > 6.5 else 0 for i in data['quality']]

# Check for missing values
missing_values_count = data.isnull().sum()
print('Missing values: \n', missing_values_count)

# Create pipeline
X = data.drop(['quality'], axis=1)
y = data['quality']

steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)

# Data into variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

pipeline.fit(X_train, y_train)

# Apply the confusion matrix
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm, '\nAccuracy score: ', accuracy_score(y_test, y_pred))

# k-Cross validation
accuracies = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=5)
print('Accuracy:', round(accuracies.mean()*100, 2))
print('Standard deviation:', round(accuracies.std()*100, 2))

# Create grid and test for the best model params
params = [{'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'knn__leaf_size': [10, 20, 30, 40, 50], 'knn__metric': ['euclidean']},
          {'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'knn__leaf_size': [10, 20, 30, 40, 50], 'knn__metric': ['manhattan']},
          {'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'knn__leaf_size': [10, 20, 30, 40, 50], 'knn__metric': ['chebyshev']},
          {'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'knn__leaf_size': [10, 20, 30, 40, 50], 'knn__p': [1, 2], 'knn__metric': ['minkowski']}]

grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=-1)

# Train the model
grid.fit(X_train, y_train)

best_acc = grid.best_score_
best_params = grid.best_params_

print("Best accuracy: ", best_acc, "\nBest parameters: ", best_params)

# Score the test datas
test_data_result = grid.score(X_test, y_test)
print('Test data result: ', test_data_result)

# The model achieves ~87.5% accuracy
