import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "freetime",]]
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
acc1 = int(acc * 100)

predictions = linear.predict(x_test)
sn = 0
for x in range(len(predictions)):
    sn += 1
    print(f"student nr. {sn}")
    print(f"Predicted grade: {predictions[x]}\nData: {x_test[x]}\nFinal grade: {y_test[x]}\n\n")

print(f"The prediction had {acc1}% accuracy\n")
