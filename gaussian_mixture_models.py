from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    if initial_means==None:
        initial_means=[]
        for i in range(k):
            row=np.random.randint(0,image_values.shape[0])
            col=np.random.randint(0,image_values.shape[1])
            initial_means.append(image_values[row][col])
        initial_means=np.asarray(initial_means,dtype=np.float32)
    while True:
        diff=[]
        flat_image=flatten_image_matrix(image_values)
        for i in range(k):
            diff.append(np.sum(np.square(flat_image-initial_means[i]),axis=1))
        cluster=np.argmin(diff,axis=0)
        new_mean=[np.asarray([0 for x in range(image_values.shape[2])])for y in range(k)]
        for i in range(len(cluster)):
            new_mean[cluster[i]]=np.add(new_mean[cluster[i]],flat_image[i])
        from collections import Counter
        pop=Counter(cluster)
        pop=np.asarray([pop[i] for i in range(k)])
        for i in range(k):
            new_mean[i]=new_mean[i]/pop[i]
        if np.array_equal(new_mean,initial_means):
            for i in range(len(cluster)):
                flat_image[i]=new_mean[cluster[i]]
            return unflatten_image_matrix(flat_image,image_values.shape[1])
        else:
            initial_means=new_mean



def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.gamma=None
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """

        N=np.multiply(1/(np.sqrt(2*np.pi)*np.sqrt(self.variances)),np.exp(np.square(np.subtract(val,self.means))/-2*self.variances))
        px=np.multiply(self.mixing_coefficients,N)
        return round(np.log(np.sum(px)),4)

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        for i in range(self.num_components):
            x=np.random.randint(0,self.image_matrix.shape[0])
            y=np.random.randint(0,self.image_matrix.shape[1])
            self.means[i]=self.image_matrix[x][y]
        self.variances.fill(1)
        self.mixing_coefficients.fill(float(1)/self.num_components)

    def train_model(self, convergence_function=default_convergence):
        convergence_ctr=0
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        #expectation stage
        dupl=self.image_matrix.flatten().copy()[:,np.newaxis]
        gamma=[]
        while True:
            prev_likelihood=self.likelihood()
            lnN_term=-np.log(self.variances*2*np.pi)*0.5
            sub_den=2*self.variances
            #expectation step
            sub_term=np.square(np.subtract(dupl,self.means))/sub_den
            N=np.subtract(lnN_term,sub_term)
            px=N+np.log(self.mixing_coefficients)
            px=np.exp(px)
            gamma=np.divide(px,np.sum(px,axis=1)[:,np.newaxis])
            sum_gamma=np.sum(gamma,axis=0)

            #maximization stage
            self.mixing_coefficients=np.sum(gamma,axis=0)/len(dupl)
            self.means=np.divide(np.sum(np.multiply(gamma,dupl),axis=0),sum_gamma)
            self.variances=np.sum(np.multiply(np.square(np.subtract(dupl,self.means)),gamma),axis=0)/sum_gamma
            present_likelihood=self.likelihood()
            convergence_ctr,converged=convergence_function(prev_likelihood,present_likelihood,convergence_ctr)
            if converged:
                print self.means
                self.gamma=gamma
                return

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.
        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        dupl=self.image_matrix.flatten()
        for x in range(len(self.gamma)):
            dupl[x]=self.means[np.argmax(self.gamma[x])]
        return dupl.reshape(self.image_matrix.shape)

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        dupl=self.image_matrix.flatten()[:,np.newaxis]
        const=1/(2*np.pi)**0.5
        N=np.sum(np.multiply(self.mixing_coefficients,(np.multiply(const/np.sqrt(self.variances),np.exp(np.divide(np.multiply(np.square(np.subtract(dupl,self.means)),-1),np.multiply(2,self.variances)))))),axis=1)
        return np.sum(np.log(N))


    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        self.initialize_training()
        best_seg=None
        best_like=self.likelihood()
        for i in range(iters):
            self.train_model()
            pr_likelihood=self.likelihood()
            if pr_likelihood>best_like:
                best_seg=self.segment()
                best_like=pr_likelihood
        return best_seg



