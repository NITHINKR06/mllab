import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = np.array([5, 10, 15, 20, 25, 30]).reshape(-1, 1)
y = np.array([3, 4, 6, 7, 9, 11])

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

print("Intercept:", round(model.intercept_, 2))
print("Slope:",     round(model.coef_[0], 2))
print("MAE:",  round(mean_absolute_error(y, y_pred), 2))
print("MSE:",  round(mean_squared_error(y, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 2))
print("R2:",   round(r2_score(y, y_pred), 2))

# Plot
plt.scatter(x, y, color='red', label='Actual Data')
plt.plot(x, y_pred, color='blue', label='Regression Line')
plt.xlabel("Open Ports")
plt.ylabel("Incidents")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()




#----------------------------------------------------------------------------------------

## knn

import numpy as np
from collections import Counter

X_train = np.array([[2,150],[3,200],[5,250],[6,300],[7,350]])
y_train = np.array([0, 0, 1, 1, 1])  # 0=Normal, 1=Malicious
k = 3

def knn(test_point):
    distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
    k_labels  = y_train[np.argsort(distances)[:k]]
    return Counter(k_labels).most_common(1)[0][0]

for point in [[3,180], [4,230], [6,320]]:
    result = knn(point)
    print(f"Traffic {point} --> {'MALICIOUS' if result == 1 else 'NORMAL'}")



#-----------------------------------------------------------------------------------------

#Naive Bayes

import pandas as pd

df = pd.DataFrame({
    'Offer': ['Yes','Yes','No','No','Yes','No','Yes','No'],
    'Money': ['Yes','No','Yes','No','Yes','No','Yes','Yes'],
    'Spam':  ['Yes','Yes','No','No','Yes','No','No','Yes']
})

# Test email
offer, money = 'Yes', 'Yes'

for label in ['Yes', 'No']:
    subset = df[df['Spam'] == label]
    prob   = (len(subset) / len(df)) * \
             (len(subset[subset['Offer'] == offer]) / len(subset)) * \
             (len(subset[subset['Money'] == money]) / len(subset))
    print(f"P(Spam={label} | X) = {prob:.4f}")

spam    = df[df['Spam']=='Yes']
notspam = df[df['Spam']=='No']

p_spam    = (len(spam)/len(df))    * (len(spam[spam['Offer']==offer])/len(spam))       * (len(spam[spam['Money']==money])/len(spam))
p_notspam = (len(notspam)/len(df)) * (len(notspam[notspam['Offer']==offer])/len(notspam)) * (len(notspam[notspam['Money']==money])/len(notspam))

print("\nResult:", "SPAM" if p_spam > p_notspam else "NOT SPAM")
