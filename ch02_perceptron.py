# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt












# define the Perceptron class
class Perceptron:
    
    """ Perceptron classifier.

    Parameters
    ----------
    eta: Learning rate (between 0.0 and 1.0)
    n_iter: Passes over the training dataset.
    random_state: Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_: 1d-array
        Weights after fitting.
    b_: Scalar
        Bias unit after fitting.
    errors_: List
        Number of misclassifications (updates) in each epoch.

    """
    
    def __init__ (self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        
        Parameters        
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y: array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self: object
        """

        # create a random number generator with the specified seed
        rgen = np.random.RandomState(self.random_state)
        
        # initialize weights to small random numbers
        # drawn from a normal distribution 
        # with mean 0 and standard deviation 0.01
        self.w_ = rgen.normal(loc=0.0, 
                              scale=0.01, 
                              size=X.shape[1] # number of features

                              # shape represents the number of samples 
                              # and the number of features in the training data

                              )
        
        # initialize bias to 0
        self.b_ = np.float_(0.)

        # initialize errors list to keep track of misclassifications in each epoch
        # this will be used to plot the number of misclassifications over epochs
        # epoch means one pass through the entire training dataset
        self.errors_ = []

        # update weights and bias for each training sample
        # for each epoch, we loop through all training samples and update the weights and bias

        # the update is based on the difference between the target value and the predicted value

        # the target value is the actual class label for the training sample, 
        # and the predicted value is the output of the perceptron for that sample

        for _ in range(self.n_iter):
            # initialize the number of misclassifications to 0 for this epoch
            errors = 0

            # loop through each training sample and its corresponding target value
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
        
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_ # dot product of X and w_ + b_
    
    # apply the unit step function to the net input to get the predicted class label
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0) # if net input is >= 0.0, return 1, else return -1
    












# extract the first 100 rows of the iris dataset and the class labels for setosa and versicolor


# read the iris dataset from a URL and load it into a pandas DataFrame
s = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(s, header=0, encoding='utf-8')
# print(df.tail())


# select the first 100 rows and the column at index 4 (class label) as the target vector y
y = df.iloc[0:100, 4].values

# convert the class labels from strings to integers
# setosa is mapped to 0 and versicolor is mapped to 1
y = np.where(y=='setosa', 0, 1)


# select the first 100 rows and 
# the columns at index 0 and 2 (sepal length and petal length) as the feature matrix X
X = df.iloc[0:100, [0, 2]].values













# plot the data points for the two classes using different colors and markers

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()
# plt.close()














# create an instance of the Perceptron class with a learning rate of 0.1 and 10 iterations


ppn = Perceptron(eta=0.1, n_iter=10)

# fit the model to the training data
ppn.fit(X, y)

# plot the number of misclassifications in each epoch

# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassifications')
# plt.show()














# visualize the decision boundary of the trained perceptron model

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # create a mesh grid of points with the specified resolution
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # predict the class labels for each point in the mesh grid
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # reshape the predicted class labels to match the shape of the mesh grid
    lab = lab.reshape(xx1.shape)

    # plot the decision surface using a filled contour plot
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    # set the limits of the plot
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot the original data points on top of the decision surface
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0],
                    y = X[y == cl, 1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label = f'Class {cl}',
                    edgecolor = 'black')













# call the function to plot the decision regions for the trained perceptron model
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()