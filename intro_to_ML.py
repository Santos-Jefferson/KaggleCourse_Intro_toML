# Importing libraries for our code
import pandas as pd

# dataset filename
dataset = "melb_data.csv"

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    function to get MAE (Mean Absolute Error) based on max_leaf_nodes of the model
    :param max_leaf_nodes: integer
    :param train_X: dataframe
    :param val_X: dataframe
    :param train_y: series
    :param val_y: series
    :return: return the MAE of that max_leaf_nodes inserted
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)

# using pandas to read the .csv file
df = pd.read_csv(dataset)

# summary stats of the dataframe(df)
print(df.describe())

# checking only the column names of the df
print(df.columns)

# checking how many null values we have in our df
print(df.isnull().sum())

# cleaning up the df, dropping NA rows (axis=0)
df = df.dropna(axis=0)

# checking summary stats after the cleaning up
print(df.describe())

# defining our target column (dependent variable we want to predict)
y = df.Price
print(y)

# correlation matrix among the columns (target is price column)
print(df.corr())

# creating a df with specific columns only
df_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# creating ou training data (usually X letter)
X = df[df_features]
print(X.describe())
print(X.head())

# instantiating the model and fitting the data (creating the regression formula, let's say)
from sklearn.tree import DecisionTreeRegressor

df_model = DecisionTreeRegressor(random_state=1)
df_model.fit(X, y)

# using the training data to predict results (it is not correct, just for learning purposes)
print("Making predictions for the next 5 houses")
print(X.head())
print("The predictions are...")
print(df_model.predict(X.head()))
print("The actual values are...")
print(y.head())

# model validation with MAE (plain English: "On average, our predictions are off by about X.")
# MAE get the residual values (predicted - actual) and take the average of it.
from sklearn.metrics import mean_absolute_error
predicted_home_prices = df_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# spliting the data between training and testing
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
df_model = DecisionTreeRegressor(max_leaf_nodes=544)
df_model.fit(train_X, train_y)
val_predictions = df_model.predict(val_X)

print("Decision Tree Class MAE:", mean_absolute_error(val_y, val_predictions))

# calling here the function to find the best number for max_leaf_nodes of the model
for max_leaf_nodes in range(500, 560, 2):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: %d \t\t MAE: %d" % (max_leaf_nodes, my_mae))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Now using another regressor at the same data to find a best MAE
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("Random Forest Class MAE:", mean_absolute_error(val_y, melb_preds))