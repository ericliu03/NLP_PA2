from classifier import Classifier
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as optimizer
from collections import Counter


class MaximumEntropy(Classifier):
    """A Maximum Entropy Classifier."""
    def __init__(self, min_f=50, max_f=500):
        self.final_lambdas = []
        self.number_of_features = Counter()
        self.raw_data = []
        self.number_of_classes = Counter()
        self.indices = {}
        self.ps = {}
        self.min_f = min_f
        self.max_f = max_f
        self.special_x = object()

    def get_model(self):
        return 0

    def set_model(self, model):
        pass

    model = property(get_model, set_model)

    def get_index(self, x, y):
        """give/get an index k of a tuple (x, y)"""
        if (x, y) not in self.indices:
            self.indices[(x, y)] = len(self.indices)
        return self.indices[(x, y)]

    def get_indices(self, x_list, y):
        """transfer word and label to indices"""
        return_list = [self.indices[(self.special_x, y)]]
        for x in x_list:
            if x not in self.number_of_features:
                continue
            return_list.append(self.indices[(x, y)])
        return np.array(return_list)

    def calculate_p(self, x_list, y, lambdas):
        """calculate p for a x, y pair with argument lambda"""
        store_key = tuple([y]+x_list)
        if store_key in self.ps:
            return self.ps[store_key]
        else:
            numerator = np.exp(np.sum(lambdas[self.get_indices(x_list, y)]))
            denominator = numerator
            for y_dash in self.number_of_classes:
                if y_dash != y:
                    denominator += np.exp(np.sum(lambdas[self.get_indices(x_list, y_dash)]))
            self.ps[store_key] = numerator/denominator
        return numerator/denominator

    def calculate_log_likelihood(self, lambdas):
        """calculate log likelihood for every data (x, y) given, return -log in order to minimize"""
        log_sum = .0
        self.ps.clear()
        for instance in self.raw_data:
            label = instance[0]
            data = instance[1]
            log_sum += np.log(self.calculate_p(data, label, lambdas))
        return -log_sum

    def calculate_gradient(self, lambdas):
        """calculate gradient for each lambda and return the lambda vector"""
        observed_vector = np.zeros(len(lambdas))
        expected_vector = np.zeros(len(lambdas))

        for instance in self.raw_data:
            data = instance[1]
            label = instance[0]
            observed_vector[self.get_indices(data, label)] += 1
            for y_dash in self.number_of_classes:
                expected_vector[self.get_indices(data, y_dash)] += self.calculate_p(data, y_dash, lambdas)
        return - (observed_vector - expected_vector)

    def collect_x_y(self, instances, min_f, max_f):
        """collect every kind of x and y"""
        self.get_data_set(instances)
        self.eliminate_features(min_f, max_f)
        self.give_index()

    def get_data_set(self, instances):
        """store every instance and collect each kind of features"""
        self.raw_data = instances
        for instance in instances:
            self.number_of_classes[instance.label] += 1
            for feature in instance.features():
                self.number_of_features[feature] += 1

    def eliminate_features(self, min_f, max_f):
        """delete features that appear certain times"""
        temp = self.number_of_features.copy()
        for key, value in temp.iteritems():
            if value < min_f or value > max_f:
                del self.number_of_features[key]

        temp_feature = []
        temp_data = []

        for instance in self.raw_data:
            for x in instance.features():
                if x in self.number_of_features:
                    temp_feature.append(x)
            temp_data.append([instance.label, temp_feature])
            temp_feature = []

        self.raw_data = temp_data[:]

    def give_index(self):
        """give each combination of x, y a index k"""
        for y_key in self.number_of_classes.iterkeys():
            self.get_index(self.special_x, y_key)
            for x_feature in self.number_of_features.iterkeys():
                self.get_index(x_feature, y_key)

    def calculate_log_likelihood_for_classify(self, instance, lambdas):
        """calculate log likelihood for one specific instance"""
        self.ps = {}
        log_likelihood = {}
        for y in self.number_of_classes:
            if y in log_likelihood:
                log_likelihood[y] += np.log(self.calculate_p(instance.features(), y, lambdas))
            else:
                log_likelihood[y] = np.log(self.calculate_p(instance.features(), y, lambdas))
        return log_likelihood

    def get_total_number(self):
        return len(self.number_of_classes) * (len(self.number_of_features) + 1)

    def train(self, instances):
        self.collect_x_y(instances, self.min_f, self.max_f)
        initial_lambda = np.zeros(self.get_total_number())
        returns = optimizer(self.calculate_log_likelihood, x0=initial_lambda,
                            fprime=self.calculate_gradient, iprint=1, maxiter=100)
        self.final_lambdas = returns[0]

    def classify(self, instance):
        return_dict = self.calculate_log_likelihood_for_classify(instance, self.final_lambdas)
        return max([(value, key) for key, value in return_dict.iteritems()])[1]


class MaximumEntropyWithPrior(MaximumEntropy):
    def __init__(self, min_f=50, max_f=800, sigma=10):
        super(MaximumEntropyWithPrior, self).__init__(min_f=min_f, max_f=max_f)
        self.sigma = sigma

    def calculate_log_likelihood(self, lambdas):
        log_sum = .0
        self.ps.clear()
        for instance in self.raw_data:
            label = instance[0]
            data = instance[1]
            log_sum += np.log(self.calculate_p(data, label, lambdas))
        log_sum += lambdas*lambdas/(self.sigma*self.sigma)
        return -log_sum

    def calculate_gradient(self, lambdas):
        observed_vector = np.zeros(len(lambdas))
        expected_vector = np.zeros(len(lambdas))

        for instance in self.raw_data:
            data = instance[1]
            label = instance[0]
            observed_vector[self.get_indices(data, label)] += 1
            for y_dash in self.number_of_classes:
                expected_vector[self.get_indices(data, y_dash)] += self.calculate_p(data, y_dash, lambdas)
        posterior = 2*lambdas/(self.sigma*self.sigma)
        return -(observed_vector - expected_vector - posterior)


