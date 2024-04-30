import pandas as pd
import numpy as np
import math
from SVM import *
import random
import sklearn
import matplotlib.pyplot as plt



df = pd.read_excel('breast_cancer_wisconsin.xlsx')
data = df.to_numpy()
mfull,n = data.shape


index = [x for x in range(mfull)]

m = math.ceil(mfull/5)


maxiter = 1
Error_rate = np.zeros((maxiter,1))

for iter in range(maxiter):
    print(iter)
    I = random.sample(index,m)
    
    J = [i for i in index if i not in I]

    trainingdata = data[J,:]

    Xt = trainingdata[:,0:n-2]
    Yt = trainingdata[:,-1]

    testdata = data[I,:]

    X = testdata[:,0:n-2]
    Y = testdata[:,-1]

    [v,b,y] = soft_svm(Xt,Yt)

    Z = np.sign(np.matmul(X,v) + b*np.ones((m,1)))

    num = 0
    for i in range(m):
        if Z[i] != Y[i]:
            num = num + 1

    Error_rate[iter] = (num/m) * 100
print(Error_rate)
Average_accuracy = 100 - np.mean(Error_rate)
print(Average_accuracy)

plt.boxplot(100-Error_rate)
plt.show()

