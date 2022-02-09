import scipy.io
from os import path
import numpy
import math
# import geneNewData

def read_data(file_name):
    dataset_values = scipy.io.loadmat("resources/" + file_name)
    return dataset_values


def data_processing():
    train_0_data = read_data("train_0_img.mat")
    train_0_d = train_0_data.get('target_img')
    print("train_0", len(train_0_d))
    # train_0_label = read_data("train_0_label.mat")
    # train_0_l = train_0_label.get('target_img')
    # # print(train_0_label, train_0_l)
    # print(len(train_0_l))

    train_1_data = read_data("train_1_img.mat")
    train_1_d = train_1_data.get('target_img')
    # print(train_1_data, "\n", train_1_d)
    print("train_1", len(train_1_d))
    # train_1_label = read_data("train_1_label.mat")
    # train_1_l = train_1_label.get('target_img')
    # # print(train_1_label, "\n", train_1_l)
    # print(len(train_1_l))

    test_0_data = read_data("test_0_img.mat")
    test_0_d = test_0_data.get('target_img')
    # print(test_0_data, "\n", test_0_d)
    print("test_0", len(test_0_d))

    test_1_data = read_data("test_1_img.mat")
    test_1_d = test_1_data.get('target_img')
    # print(test_1_data, "\n", test_1_d)
    print("test_1", len(test_1_d))

    # print(train_0_data, "\n", train_0_d)
    print(len(train_0_d[0]), len(train_0_d[0][0]))
    print(train_0_d[0])
    feature1_train0 = numpy.array([numpy.mean(image0) for image0 in train_0_d])
    feature2_train0 = numpy.array([numpy.std(image0) for image0 in train_0_d])
    # train_0_T = train_0_d.reshape(28, 165844)
    # feature1_train0_T = numpy.array([numpy.mean(image0) for image0 in train_0_T])
    print(feature1_train0, len(feature1_train0), feature2_train0, len(feature2_train0))
    # print(feature1_train0_T, len(feature1_train0_T))


# Provided Code
# My ASU ID - 1220905569
def main():
    myID = '5569'
    # geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train' + myID + '.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train' + myID + '.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset' + '.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset' + '.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0), len(train1), len(test0), len(test1)])
    print('Your trainset and testset are generated successfully!')

    # Code starts
    print("\n\n")
    # print(Numpyfile0)
    # print(train0[0]) # print first image
    print("print size of first image")
    print(len(train0[0]), len(train0[0][0]))  # print size of first image

    # Start of task 1
    print("\n\n")
    # without numpy.array -> type is <class 'list'>
    # with numpy.array -> type is <class 'numpy.ndarray'>
    feature1_train0 = numpy.array([numpy.mean(image) for image in train0])
    feature2_train0 = numpy.array([numpy.std(image) for image in train0])
    print("Size of feature1 and feature2 for train 0")
    print(len(feature1_train0), len(feature2_train0), type(feature1_train0), type(feature2_train0))

    feature1_train1 = numpy.array([numpy.mean(image) for image in train1])
    feature2_train1 = numpy.array([numpy.std(image) for image in train1])
    print("Size of feature1 and feature2 for train 1")
    print(len(feature1_train1), len(feature2_train1), type(feature1_train1), type(feature2_train1))

    feature1_test0 = numpy.array([numpy.mean(image) for image in test0])
    feature2_test0 = numpy.array([numpy.std(image) for image in test0])
    print("Size of feature1 and feature2 for test 0")
    print(len(feature1_test0), len(feature2_test0), type(feature1_test0), type(feature2_test0))
    # print(feature1_test0, type(feature1_test0))

    feature1_test1 = numpy.array([numpy.mean(image) for image in test1])
    feature2_test1 = numpy.array([numpy.std(image) for image in test1])
    print("Size of feature1 and feature2 for test 1")
    print(len(feature1_test1), len(feature2_test1), type(feature1_test1), type(feature2_test1))
    # End of task 1

    # Start of task 2
    print("\n\n")
    # train 0 and 1
    feature1_mean_train0 = numpy.mean(feature1_train0)
    feature1_var_train0 = numpy.var(feature1_train0)

    feature2_mean_train0 = numpy.mean(feature2_train0)
    feature2_var_train0 = numpy.var(feature2_train0)

    feature1_mean_train1 = numpy.mean(feature1_train1)
    feature1_var_train1 = numpy.var(feature1_train1)

    feature2_mean_train1 = numpy.mean(feature2_train1)
    feature2_var_train1 = numpy.var(feature2_train1)

    # test 0 and 1
    feature1_mean_test0 = numpy.mean(feature1_test0)
    feature1_var_test0 = numpy.var(feature1_test0)

    feature2_mean_test0 = numpy.mean(feature2_test0)
    feature2_var_test0 = numpy.var(feature2_test0)

    feature1_mean_test1 = numpy.mean(feature1_test1)
    feature1_var_test1 = numpy.var(feature1_test1)

    feature2_mean_test1 = numpy.mean(feature2_test1)
    feature2_var_test1 = numpy.var(feature2_test1)

    print("feature1_mean_train0", feature1_mean_train0)
    print("feature1_var_train0", feature1_var_train0)
    print("feature2_mean_train0", feature2_mean_train0)
    print("feature2_var_train0", feature2_var_train0)
    print("feature1_mean_train1", feature1_mean_train1)
    print("feature1_var_train1", feature1_var_train1)
    print("feature2_mean_train1", feature2_mean_train1)
    print("feature2_var_train1", feature2_var_train1)

    print("feature1_mean_test0", feature1_mean_test0)
    print("feature1_var_test0", feature1_var_test0)
    print("feature2_mean_test0", feature2_mean_test0)
    print("feature2_var_test0", feature2_var_test0)
    print("feature1_mean_test1", feature1_mean_test1)
    print("feature1_var_test1", feature1_var_test1)
    print("feature2_mean_test1", feature2_mean_test1)
    print("feature2_var_test1", feature2_var_test1)
    # End of task 2

    # Start of task 3
    print("\n\n")
    # P(Y=0|X) = ( P(X|Y=0) * P(Y=0) )/P(X)
    # P(Y=0) = .5

    # Image 0
    # P(X|Y=0)
    PX1_givenY0 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature1_var_train0))) * numpy.exp(
        (-1 * (feature1_value - feature1_mean_train0)**2)/(2 * feature1_var_train0))) for feature1_value in
                   feature1_test0])
    PX2_givenY0 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature2_var_train0))) * numpy.exp(
        (-1 * (feature2_value - feature2_mean_train0)**2)/(2 * feature2_var_train0))) for feature2_value in
                   feature2_test0])
    # print("PX1_givenY0: ", PX1_givenY0, "PX2_givenY0: ", PX2_givenY0)
    PX_0_givenY0 = numpy.multiply(PX1_givenY0, PX2_givenY0)

    # P(X|Y=1)
    PX1_givenY1 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature1_var_train1))) * numpy.exp(
        (-1 * (feature1_value - feature1_mean_train1) ** 2) / (2 * feature1_var_train1))) for feature1_value in
                   feature1_test0])
    PX2_givenY1 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature2_var_train1))) * numpy.exp(
        (-1 * (feature2_value - feature2_mean_train1) ** 2) / (2 * feature2_var_train1))) for feature2_value in
                   feature2_test0])
    # print("PX1_givenY1: ", PX1_givenY1, "PX2_givenY1: ", PX2_givenY1)
    PX_0_givenY1 = numpy.multiply(PX1_givenY1, PX2_givenY1)

    # print(PX0_givenY0, PX0_givenY1)
    print(type(PX_0_givenY0), type(PX_0_givenY1))
    test0_prediction = PX_0_givenY0 > PX_0_givenY1
    print("test0_prediction: ", test0_prediction, type(test0_prediction))

    # Image 1
    # P(X|Y=0)
    PX1_givenY0 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature1_var_train0))) * numpy.exp(
        (-1 * (feature1_value - feature1_mean_train0) ** 2) / (2 * feature1_var_train0))) for feature1_value in
                   feature1_test1])
    PX2_givenY0 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature2_var_train0))) * numpy.exp(
        (-1 * (feature2_value - feature2_mean_train0) ** 2) / (2 * feature2_var_train0))) for feature2_value in
                   feature2_test1])
    # print("PX1_givenY0: ", PX1_givenY0, "PX2_givenY0: ", PX2_givenY0)
    PX_1_givenY0 = numpy.multiply(PX1_givenY0, PX2_givenY0)

    # P(X|Y=1)
    PX1_givenY1 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature1_var_train1))) * numpy.exp(
        (-1 * (feature1_value - feature1_mean_train1) ** 2) / (2 * feature1_var_train1))) for feature1_value in
                   feature1_test1])
    PX2_givenY1 = numpy.array([((1 / (numpy.sqrt(2 * math.pi * feature2_var_train1))) * numpy.exp(
        (-1 * (feature2_value - feature2_mean_train1) ** 2) / (2 * feature2_var_train1))) for feature2_value in
                   feature2_test1])
    # print("PX1_givenY1: ", PX1_givenY1, "PX2_givenY1: ", PX2_givenY1)
    PX_1_givenY1 = numpy.multiply(PX1_givenY1, PX2_givenY1)

    # print(PX_1_givenY0, PX_1_givenY1)
    print(type(PX_1_givenY0), type(PX_1_givenY1))
    test1_prediction = PX_1_givenY0 > PX_1_givenY1
    print("test1_prediction: ", test1_prediction, type(test1_prediction))
    # End of task 3

    # Start of task 4
    print("\n\n")
    test0_accuracy = numpy.where(test0_prediction == True)
    test0_accuracy_perc = len(test0_accuracy[0]) / 980
    # print("test0_accuracy: ", test0_accuracy, type(test0_accuracy))
    print("test0_accuracy_perc: ", test0_accuracy_perc, type(test0_accuracy_perc))

    test1_accuracy = numpy.where(test1_prediction == False)
    test1_accuracy_perc = len(test1_accuracy[0]) / 1135
    # print("test1_accuracy: ", test1_accuracy, type(test1_accuracy))
    print("test1_accuracy_perc: ", test1_accuracy_perc, type(test1_accuracy_perc))

    # ~92 % accuracy.(digit0:  899 / 980, digit1:  1048 / 1135)
    # End of Task 4

    print("Final result in an array")
    final_array = [feature1_mean_train0, feature1_var_train0, feature2_mean_train0, feature2_var_train0,
                   feature1_mean_train1, feature1_var_train1, feature2_mean_train1, feature2_var_train1,
                   test0_accuracy_perc, test1_accuracy_perc]
    print(final_array)

'''
Question 1
Suppose that a set of samples x_1, x_2, ..., x_n , all real numbers, are drawn i.i.d. from the same distribution. 
Also assume that this distribution is a Gaussian distribution, which can be represented as N(mu, sigma) 
Write a function that accepts a set of samples and returns the MLE estimator for mu.
 '''
def mle(samples):
    sum = 0
    for elem in samples:
        sum += elem

    return sum / len(samples)


if __name__ == '__main__':
    basepath = path.dirname(__file__)
    print(basepath)

    generic_resource_path = path.abspath(path.join(basepath, 'resources')) + "/"
    print(generic_resource_path)

    data_processing()
