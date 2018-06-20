import matplotlib.pyplot as plt
from sklearn import svm, datasets

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001)


"train classifier"
clf.fit(digits.data[:-10], digits.target[:-10])

"predict & print result"
print("Prediction:", clf.predict(digits.data[-9]))
print("Target:", digits.target[-9])

"Plot digit"
plt.imshow(digits.images[-9], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
