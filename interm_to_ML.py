# Importing libraries for our code
import pandas as pd
from sklearn.model_selection import train_test_split

# dataset filename
dataset = "melb_data.csv"

# using pandas to read the .csv file
df = pd.read_csv(dataset)

# defining our target column (dependent variable we want to predict)
y = df.Price

# selecting the features dropping the target (Price) column
melb_predictors = df.drop(['Price'], axis=1)

# selecting only numerical predictors for now
X = melb_predictors.select_dtypes(exclude=['object'])

# spliting the data between training and testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function to compare different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# approach 1 - drop columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from approach 1:", score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# approach 2 - imputation
from sklearn.impute import SimpleImputer

# imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_valid.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from approach 2:", score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# approach 3
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

#imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from approach 3:", score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# shape of training data
print(X_train.shape)

# number of missing values in each column of training data
missing_val_count_by_column = X_train.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column > 0])