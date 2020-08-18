import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read a simple Salary x experience DataSet
df = pd.read_csv("datasets_8634_12080_Salary_Data.csv")

# Split the values into 2 arrays
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Splits the arrays into train(80%) and test(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the linear regression algorithm
linear_regression = LinearRegression()
linear_regression.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Predict the values based on the regression algorithm
y_predict = linear_regression.predict(x_test.reshape(-1, 1))


# Plot points and fit line for training data
plt.scatter(x_train, y_train, color='teal', edgecolors='black', label='Training-set observation points')
plt.plot(x_train.reshape(-1, 1), linear_regression.predict(x_train.reshape(-1, 1)), color='grey', label='Fit Regression Line')
plt.title('Salary vs Experience')
plt.xlabel('Experience (in years)')
plt.ylabel('Salary (in USD)')

# plot scatter points and line for test data
plt.scatter(x_test, y_test, color='red', edgecolors='black', label='Test-set observation points')
plt.legend()
plt.show()
