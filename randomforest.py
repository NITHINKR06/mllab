import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = {
    'Traffic': [100, 200, 150, 300, 250, 400, 120, 350, 500, 450],
    'Login_Attempts': [1, 5, 2, 10, 7, 12, 1, 9, 15, 11],
    'Suspicious_IP': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    'Attack': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[['Traffic', 'Login_Attempts', 'Suspicious_IP']]
y = df['Attack']

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, random_state=42
)

def bootstrap_sample(X, y):
    n = len(X)
    idx = np.random.choice(n, n, replace=True)
    return X.iloc[idx], y.iloc[idx]

n_trees = 5
trees = []

for _ in range(n_trees):
    X_s, y_s = bootstrap_sample(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_s, y_s)
    trees.append(tree)
    
def predict(X):
    preds = np.array([tree.predict(X) for tree in trees])
    preds = np.swapaxes(preds, 0, 1)
    
    final = []
    for p in preds:
        final.append(Counter(p).most_common(1)[0][0])
        
    return np.array(final)

y_pred = predict(X_test)
acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", acc)

new_data = pd.DataFrame([[280, 8, 1]],
                        columns=['Traffic', 'Login_Attempts', 'Suspicious_IP'])

print("Prediction", predict(new_data)[0])
    