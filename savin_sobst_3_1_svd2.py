# Метод главных компонент Урок 5 23.20

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
    
# формирование вектора

x=np.arange(1,11)*100
y=2*x+np.random.randn(10)*5  # шумы
X=np.vstack((x,y))
print(X)

plt.scatter(X[0],X[1])
plt.show()

# 24.25 цетрирование
Xcentered=(X[0]-x.mean(),X[1]-y.mean())
m=(x.mean(), y.mean())
print(Xcentered)
print("Mean вектор: ",m)

plt.scatter(Xcentered[0],Xcentered[1])
plt.show()

# 25.10 Матрица ковариации
covmat=np.cov(Xcentered)
print(covmat,"\n")
print("Variance of X ",np.cov(Xcentered)[0,0])
print("Variance of Y ",np.cov(Xcentered)[1,1])
print("Covariance of X Y",np.cov(Xcentered)[0,1])

# 25.50 нахождение собственных векторов
vecs=np.linalg.eig(covmat)
v=-vecs[1]                  # вот здесь непонятно
Xnew=np.dot(v,Xcentered)
print("Собственный вектор:\n ",Xnew)
#=======================
print(vecs[-1])
#======================= из sklearn

pca=PCA(1)
XPCAreduced=pca.fit_transform(np.transpose(X))
for xn,x_pca in zip(Xnew,XPCAreduced):
    print(xn,'-',x_pca)

