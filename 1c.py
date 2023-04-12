import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
# Declare training dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1,1,1,1])

# Training the model SVM 
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X,y)
# Draw classification boundaries 
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-0.5, 1.5)
yy = a * xx - (clf.intercept_[0]) / w[1]
# Drawing data points and classification boundaries 
plt.plot(xx, yy, '-k')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classification')
plt.show()