'''
   Recognizing digits from MNIST Dataset
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

from sklearn.datasets import load_digits
from sklearn.preprocessing import Imputer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score


class DigitRecognizer:

    each_digit_max_sample = 1000
    max_train_sample = 10000
    max_test_sample = 10000
    model_dict = {}

    def __init__(self, train_file_path, test_file_path):

        # Initializing the train and test DataFrames
        self.train_dataset = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.y_train = pd.Series()
        self.test_dataset = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.Series()

        self.__read_dataset(train_file_path, test_file_path)

    def __read_dataset(self, train_file_path, test_file_path):

        # Reading Train and Test Dataset
        self.train_dataset = pd.read_csv(filepath_or_buffer=train_file_path)
        self.test_dataset = pd.read_csv(filepath_or_buffer=test_file_path)

        # Train_Data Sampling if Training DataSet is too large
        if len(self.train_dataset.index) > self.max_train_sample:

            self.train_dataset = self.__train_data_sampling()

            # Checking if sampling worked or not
            print("\nAfter Sampling :-\n")
            count = 0
            for i in range(10):
                print("No. of rows containing '%r' digit in the dataset : %r" % (i, len(self.train_dataset[self.train_dataset.iloc[:, 0] == i].index)))
                if len(self.train_dataset[self.train_dataset.iloc[:, 0] == i].index) <= self.each_digit_max_sample:
                    count += 1

            if count == 10:
                print("\nTrain Data Sampling Successful\n")
            else:
                print("\nTrain Data Sampling Unsuccessful\n")

        # Test_Data Sampling if Test DataSet is too large
        if len(self.test_dataset.index) > self.max_test_sample:
            self.test_dataset = self.__test_data_sampling()

        self.__data_preprocessing()

        # samples = [9000]
        # acc_scores = []
        #
        # for sample in samples:
        #     train_dataset = pd.read_csv(filepath_or_buffer='Dataset/mnist_train.csv')
        #     train_dataset = train_dataset.sample(n=sample).reset_index(drop=True)
        #     x_train = train_dataset[1:]
        #     y_train = x_train.ix[:, 0]
        #
        #     test_dataset = pd.read_csv(filepath_or_buffer='Dataset/mnist_test.csv')
        #     x_test = test_dataset[1:]
        #     y_test = x_test.ix[:, 0]
        #
        #     start_time = timeit.default_timer()
        #     # lr_model(x_train, y_train, x_test, y_test, sample, acc_scores)
        #     self.lr_model(sample, acc_scores)
        #     print("Time taken for %r samples : %r" % (sample, timeit.default_timer() - start_time))
        #     print()
        #
        # plt.plot(samples, acc_scores)
        # plt.show()

    def __train_data_sampling(self):

        self.new_train_data = pd.DataFrame(columns=self.train_dataset.columns)

        print("\nBefore Sampling :-\n")
        for i in range(10):
            if len(self.train_dataset[self.train_dataset.iloc[:, 0] == i].index) > self.each_digit_max_sample:
                self.new_train_data = self.new_train_data.append(self.train_dataset[self.train_dataset.iloc[:, 0] == i].sample(n=self.each_digit_max_sample).reset_index(drop=True), ignore_index=True)
            else:
                self.new_train_data = self.new_train_data.append(self.train_dataset[self.train_dataset.iloc[:, 0] == i], ignore_index=True)
            print("No. of rows containing '%r' digit in the dataset : %r" % (i, len(self.train_dataset[self.train_dataset.iloc[:, 0] == i].index)))
        print()

        return self.new_train_data

    def __test_data_sampling(self):
        return self.test_dataset.sample(n=self.max_test_sample).reset_index(drop=True)

    def __data_preprocessing(self):

        # Checking for missing values and filling with mean value
        for i, n in enumerate(list(self.train_dataset.isnull().sum())):
            if n != 0:
                self.train_dataset.fillna(value=np.mean(self.train_dataset.ix[:, i]), inplace=True)

        for i, n in enumerate(list(self.test_dataset.isnull().sum())):
            if n != 0:
                self.test_dataset.fillna(value=np.mean(self.test_dataset.ix[:, i]), inplace=True)

        # Separating the features and labels from train and test dataset
        self.x_train = self.train_dataset.iloc[:, 1:]
        print(self.x_train.tail())
        self.y_train = self.x_train.iloc[:, 0]
        print(self.y_train.tail())
        self.x_test = self.test_dataset[1:]
        self.y_test = self.x_test.ix[:, 0]

    def model_building(self, model=None):

        # TODO : Build the given model
        if model is not None:
            pass
        else:
            self.lr_model()

    def lr_model(self):
        logisticRegression = LogisticRegression()
        logisticRegression.fit(self.x_train, self.y_train)
        y_predicted = logisticRegression.predict(self.x_test)

        acc_score = accuracy_score(self.y_test, y_predicted)
        print("Accuracy Score for LR Model :", acc_score)
        self.model_dict.update({'lr' : acc_score})

    def knn_model(self):
        pass

    def kmeans_model(self):
        pass

    def svm_model(self):
        pass

    def naivebayes_model(self):
        pass

    def decision_tree(self):
        pass

    def randomforest_model(self):
        pass

    def kmeans_cluster_model(self):
        pass

    def kminibatch_cluster_model(self):
        pass

    def hierarchical_cluster(self):
        pass

    def pca_model(self):
        pass

    def lda_model(self):
        pass


digitRecognizer = DigitRecognizer('Dataset/mnist_train.csv', 'Dataset/mnist_test.csv')
digitRecognizer.model_building()
