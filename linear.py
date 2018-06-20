import matplotlib.pyplot as plt
from sklearn import svm, datasets

digits = datasets.load_digits()

clf = svm.SVC(kernel='linear')

"preferred number of datapoints to test"
test_number = 100

"train classifier"
clf.fit(digits.data[:-test_number], digits.target[:-test_number])

"predict"
prediction = clf.predict(digits.data[-test_number:])

"slice target array to same length as array containing predictions"
target = digits.target[-test_number:]

"compare each prediction with true class"
correct = 0
for i in range(len(prediction)):
    if prediction[i] == target[i]:
        correct += 1

"print result"
print(correct, "predictions out of", test_number)
