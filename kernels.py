import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import dataset
iris = datasets.load_iris()
data = iris.data[:, :2] # we only take first two features
target = iris.target

# Create an instance of SVM and train data

svc = svm.SVC(kernel='linear').fit(data, target)

# svc = svm.SVC(kernel='rbf', gamma=0.01).fit(data, target)

# svc = svm.SVC(kernel='poly').fit(data, target)

# create a mesh to plot in
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
