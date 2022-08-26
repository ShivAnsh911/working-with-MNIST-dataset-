import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

mnist=fetch_openml('mnist_784',version=1)#will download the dataset requires internet

def digit_image(a): #function to show a digit image using pyplot
    digit=np.array(a).reshape(28,28)
    plt.imshow(digit, cmap = mpl.cm.binary, interpolation="nearest")
    plt.show()

def prediction(a): #function for prediction 5 or not 5 trained below
    if sgd_clf.predict(a)==True:
        print("Digit is 5")
    else : print("Digit is not 5")

#print(mnist.keys())
#output -> dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

#seperating data and target
d,t=mnist['data'],mnist['target']


#every image is 28x28 pixels so we need to reshape the 784 pixel values given in data
digit_image(d.iloc[0])#will show image of first digit in dataset
# will show a image of 5 

#Now the target instances in t are strings, so we will have to convert it to int
#You can use type(t[0]) to check.
t=t.astype(np.uint8)

#converting both data and target to train and test sets 
#out of 70000 instances taking first 60000 for train set and rest 10000 for test set
d_train,d_test=d[:60000],d[60000:]
t_train,t_test=t[:60000],t[60000:]

#Convert the target dataset into Binary (True and False) for a digit 5
# we will make a binary classfier to check wheather a digit is 5 or not  
t_train_5=(t_train==5)
t_test_5=(t_test==5)

#Selecting SGD classifier from Scikit-Learn for the purpose, as it is capable for handling very large datasets efficiently
#by using stochastic gradient descent
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(d_train,t_train_5)

#try below code for checking the model
prediction(d[:1])
digit_image(d[:1])
prediction(d_test[:1])
digit_image(d_test[:1])
#you can check on other instances
