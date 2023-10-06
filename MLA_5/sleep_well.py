import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
xtest = np.loadtxt("data/X_test.csv", dtype=float, delimiter=',')
xtrain = np.loadtxt("data/X_train.csv", dtype=float, delimiter=',')
ytest = np.loadtxt("data/Y_test.csv", dtype=float)
ytrain = np.loadtxt("data/Y_train.csv", dtype=float)

n_train = xtrain.shape[0]
print(n_train)
#1.1
# Plot training set class frequenices
n = (ytrain.shape)[0] # n samples
unique_vals, counts = np.unique(ytrain, return_counts=True) 
counts = counts*(1/n)
fig, ax = plt.subplots()
ax.bar(unique_vals, counts)
ax.set_ylabel("Frequency")
ax.set_xlabel("Class")
ax.set_title("Training set class frequenices")

# Plot test set Class frequenices
n2 = (ytest.shape)[0] # n samples
unique_vals2, counts = np.unique(ytest, return_counts=True) 
counts2 = counts*(1/n2)
fig, ax = plt.subplots()
ax.bar(unique_vals2, counts2)
ax.set_ylabel("Frequency")
ax.set_xlabel("Class")
ax.set_title("Test set class frequenices")
#plt.show()


#1.2
#1. Apply multi-nominal logistic regression.
log_model = LogisticRegression(tol=0.0001, C=1, random_state=0, multi_class='multinomial', solver='lbfgs').fit(xtrain,ytrain)
log_train_loss = 1 - log_model.score(xtrain, ytrain)
log_test_loss = 1 - log_model.score(xtest,ytest)

#2. Apply random forests with 50, 100, and 200 trees
# fit models
rf_model_50t  = RandomForestClassifier(n_estimators=50).fit(xtrain,ytrain)
rf_model_100t = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)
rf_model_200t = RandomForestClassifier(n_estimators=200).fit(xtrain,ytrain)
# evaluatee
rf50_train_loss = 1 - rf_model_50t.score(xtrain,ytrain)
rf50_test_loss = 1 - rf_model_50t.score(xtest,ytest)
rf100_train_loss = 1 - rf_model_100t.score(xtrain,ytrain)
rf100_test_loss = 1 - rf_model_100t.score(xtest,ytest)
rf200_train_loss = 1 - rf_model_200t.score(xtrain,ytrain)
rf200_test_loss = 1 - rf_model_200t.score(xtest,ytest)


#3. Apply k-nearest-neighbor classification
random_seed = 42
np.random.seed(random_seed)
shuffled_indices = np.random.permutation(len(xtrain))
xtrain_shuffled = xtrain[shuffled_indices]
ytrain_shuffled = ytrain[shuffled_indices]

max_sqrt_k = int(math.sqrt(n_train)) # number of k's
x_splits, y_splits = np.array_split(xtrain_shuffled, max_sqrt_k), np.array_split(ytrain_shuffled, max_sqrt_k) #Creating the cross validation splits
K_list = [] # values of k
loss_list = [] # validation losses


for i in range(0, max_sqrt_k):
    k = (i+1)**2
    x_val = x_splits[i]
    y_val = y_splits[i]

    # Combine the remaining splits for training
    x_train = np.concatenate([x_splits[j] for j in range(max_sqrt_k) if j != i])
    y_train = np.concatenate([y_splits[j] for j in range(max_sqrt_k) if j != i])

    knn_model = KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
    knn_val_loss = 1 - knn_model.score(x_val, y_val) # compute loss

    K_list.append(k)
    loss_list.append(knn_val_loss)
    

# Compute test/train loss of model with best k 
smallest_loss_idx = loss_list.index(min(loss_list))
best_K = K_list[smallest_loss_idx]
best_knn_model = KNeighborsClassifier(n_neighbors=best_K).fit(xtrain, ytrain)
knn_train_loss = 1 - best_knn_model.score(xtrain,ytrain)
knn_test_loss = 1 - best_knn_model.score(xtest,ytest)

# plot with losses for values of k
fig, ax = plt.subplots()
ax.plot(K_list, loss_list)
ax.set_ylabel("Empirical Loss")
ax.set_xlabel("K")
ax.set_title("KNN cross validation plot")

results = f"""
[Logistic regression] 
train-loss: {log_train_loss}, test-loss: {log_test_loss}

[Random forest] 
(k=50) train-loss: {rf50_train_loss}, test-loss: {rf50_test_loss}
(k=100) train-loss: {rf100_train_loss}, test-loss: {rf100_test_loss}
(k=200) train-loss: {rf200_train_loss}, test-loss: {rf200_test_loss} 

[Best KNN]
K: {best_K}
train-loss: {knn_train_loss}, test-loss: {knn_test_loss}
"""

print(results)
plt.show()