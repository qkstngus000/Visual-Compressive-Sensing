from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.linalg
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy.linalg as la

class RFClassifier(BaseEstimator, ClassifierMixin):
    """
    Random feature classifier.

    This class projects inputs onto randomly generated weights and 
    classifies them using a linear classifier.

    Parameters
    ----------

    width : int
        Number of random weights the input is 
        projected onto.

    nonlinearity: callable
        Specifies the non-linear transformation that is 
        applied to the randomly transformed inputs. The call signature
        is ``fun(x)``.
    
    weight_fun: callable
        Specifies the function to generate the random weights. The call signature is 
        ``fun(width, n_features, **kwargs, random_state). ``
        Here, `width` is a positive int and specifies the 
        number of random weights. Here, `n_features` is a positive 
        int and specifies the size of each random weight. `fun`
        must return array_like with shape (width, n_features) i.e.
        each row corresponds to a random weight.

    kwargs: dict, optional
        Additional arguments to be passed to the weight
        function. If for example if the weight function has the 
        signature ```fun(width, n_features, a, b, c, random_state, )```, 
        then `kwargs` must be a dict with three parameters and their
        keywords. 
    
    bias: ndarray of shape (width,) or (1,)
        The bias term for the randomly transformed input. If (1,), same
        bias is used for all the random weights.

    clf : sklearn linear classifier object, eg: logistic regression, 
        linear SVM. Specifies the linear classifier used for 
        classification.

    random_state: int, default=None
        Used to set the seed when generating random weights.
    
    Attributes
    -------

    W_ : ndarray of shape (width, n_features)
        Random weights that are generated.

    b_ : ndarray of shape (width, ) or (1,)
        The bias term for the randomly transformed input.

    H_ : {array-like} of shape (n_samples, width)
        Transformed train input, where n_samples is the number of samples
        and width is the number of features.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function of the 
        classifier.
    
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function of the
        classifier.

    Examples
    --------

    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from estimator import RFClassifier, V1_inspired_weights

    X, y = load_digits(return_X_y=True)
    logit = LogisticRegression(solver='saga')
    relu = lambda x: np.maximum(0, x)
    kwargs = {'t': 5, 'l': 3}
    clf = RFClassifier(width=20, weight_fun=V1_inspired_weights, 
    kwargs=kwargs, bias=2, nonlinearity=relu, clf=logit, random_state=22)
    clf.fit(X, y)
    clf.score(X, y)
    
    """

    def __init__(self, width, weight_fun, bias, nonlinearity, 
    clf, kwargs=None, random_state=None):
        self.width = width
        self.nonlinearity = nonlinearity
        self.weight_fun = weight_fun
        self.bias = bias
        self.clf = clf
        self.kwargs = kwargs
        self.random_state = random_state
    
    def _fit_transform(self, X):
        """
        Project the train input onto the random weights.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of sampless
            and n_features is the number of features.

        Returns
        -------
        self 
        
        """
        # 1d or 2d
        if len(X.shape) == 2:
            n_features = X.shape[1]
        elif len(X.shape) == 3:
            n_features = X.shape[1:]
            X = X.reshape(-1, n_features[0] * n_features[1]) # flatten images
        
        if self.kwargs is not None:
            self.W_ = self.weight_fun(self.width, n_features, 
                                        **self.kwargs, seed=self.random_state)
        else:
            self.W_ = self.weight_fun(self.width, n_features, 
                            seed=self.random_state)

        self.b_ = self.bias
        self.H_ = self.nonlinearity(np.dot(X, self.W_.T) + self.b_)

    def fit(self, X, y):
        """
        Fit the model according to the given training data. 

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator
        """
        self._fit_transform(X)
        self.clf.fit(self.H_, y)
        self.coef_ = self.clf.coef_
        self.intercept_ = self.clf.intercept_
        return self

    def transform(self, X):
        """
        Project test input onto the random weights.

        Parameters
        ----------

        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of sampless
            and n_features is the number of features.

        Returns
        -------

        H : {array-like} of shape (n_samples, width)
            Transformed input, where n_samples is the number of samples
            and n_features is the number of features.
        """
        check_is_fitted(self, ["W_", "b_"])
        if len(X.shape) == 3:
            n_features = X.shape[1:]
            X = X.reshape(-1, X.shape[1] * X.shape[2])
        H = self.nonlinearity(np.dot(X, self.W_.T) + self.b_)
        return H

    def score(self, X, y):
        """
        Returns the score on the given test data and labels.

        Parameters
        ---------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        ------
        score : float
        """
        H = self.transform(X)
        check_is_fitted(self, ["coef_", "intercept_"])
        score = self.clf.score(H, y)
        return score
        

def relu(x, thrsh=0):
    """
    Rectified Linear Unit

    Parameters
    ----------

    x : {array-like} or int
        Input data
    
    thrsh: int
        threshold for firing

    Returns
    -------

    y : {array-like} or int
        Output
    """
    return np.maximum(x, thrsh)

def poly(x, power=2):
    """
    Polynomial function. Raises input to specified power.

    Parameters
    ----------

    x : {array-like} or int
        Input
    
    power: int
        Degree of the polynomial

    Returns
    -------

    y : {array-like} or int
        Output
    """
    return np.power(x, power)