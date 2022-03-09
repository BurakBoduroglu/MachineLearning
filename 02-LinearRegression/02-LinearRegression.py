import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sales_df = pd.read_excel("sales.xlsx")

print(sales_df)

months = sales_df[["Months"]]
sales = sales_df[["Sales"]]
print(months)
print(sales)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.scatter(x_train, y_train, c = "black")
plt.title("Sales for Months")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.plot(x_test, predict, linewidth = 2)