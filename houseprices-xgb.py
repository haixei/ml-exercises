import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

train = pd.read_csv('data-sets/housing-train.csv')

# Drop columns where the majority of values is null, as well as the Id column that doesn't contribute to the model
print(train.isnull().sum().sort_values(ascending=False))
train.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'Id'], axis=1, inplace=True)

# Go trough columns to see if tere's a lot of duplicates
print('Duplicates: \n', train.duplicated().sum())

# Set features & target
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)

# Divide data into variables
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                    random_state=0)

# Removing features with mostly one value, here for categorical values
cat_col = X_train.select_dtypes(include=['object']).columns
cat_rep_cols = []
for i in cat_col:
    counts = X_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X_train) * 100 > 96:
        cat_rep_cols.append(i)

cat_rep_cols = list(cat_rep_cols)

# And here for numerical values
num_col = X_train.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
num_rep_cols = []
for i in num_col:
    counts = X_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X_train) * 100 > 96:
        num_rep_cols.append(i)

num_rep_cols = list(num_rep_cols)

# Join the lists and drop the columns
all_rep_cols = num_rep_cols + cat_rep_cols
X_train = X_train.drop(all_rep_cols, axis=1)
X_test = X_test.drop(all_rep_cols, axis=1)

# PREPROCESSING
# Divide the columns into categorical and numerical lists
cat_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
            X_train[cname].dtype == "object"]
num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = cat_cols + num_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

# Repetiveness of caterogical values
cat_rep_fig = plt.figure(figsize=(18, 18))

for i in range(len(cat_cols)):
    plt.subplot(8, 5, i+1)
    sns.countplot(x=train[cat_cols[i]], data=train[cat_cols[i]].dropna())
    plt.xticks(rotation=90)

cat_rep_fig.tight_layout(pad=1.0)

# Distributions of the numerical values
num_rep_fig = plt.figure(figsize=(18, 16))

for i in range(len(num_cols)):
    plt.subplot(9, 5, i+1)
    sns.distplot(train[num_cols[i]].dropna(), kde=False)
    plt.xticks(rotation=90)

num_rep_fig.tight_layout(pad=1.0)

# Correlation between figures
num_DF = pd.DataFrame(num_cols)

plt.figure(figsize=(15, 13))
correlation = train[num_cols].corr()
sns.heatmap(correlation, mask=correlation < 0.8, linewidth=0.5, cmap='GnBu')

# Drop columns that interfere with the model
bad_cols = ['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars']
X_train.drop(bad_cols, axis=1, inplace=True)
X_test.drop(bad_cols, axis=1, inplace=True)
num_cols = [x for x in num_cols if x not in bad_cols]
cat_cols = [x for x in cat_cols if x not in bad_cols]

# Transformers
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

# CREATE MODEL
model = XGBRegressor()
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('scaling', RobustScaler()),
                              ('model', model)])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_test, preds)
print('MAE:', score)