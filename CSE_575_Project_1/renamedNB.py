
# coding: utf-8

# In[2]:


import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='5569'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    # Code starts
    print("\n\n")
    # print(Numpyfile0)
    # print(train0[0]) # print first image
    print("print size of first image")
    print(len(train0[0]), len(train0[0][0]), type(train0)) # print size of first image
    
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
#     feature1_mean_test0 = numpy.mean(feature1_test0)
#     feature1_var_test0 = numpy.var(feature1_test0)
    
#     feature2_mean_test0 = numpy.mean(feature2_test0)
#     feature2_var_test0 = numpy.var(feature2_test0)
    
#     feature1_mean_test1 = numpy.mean(feature1_test1)
#     feature1_var_test1 = numpy.var(feature1_test1)
    
#     feature2_mean_test1 = numpy.mean(feature2_test1)
#     feature2_var_test1 = numpy.var(feature2_test1)
    
    print("feature1_mean_train0", feature1_mean_train0)
    print("feature1_var_train0", feature1_var_train0)
    print("feature2_mean_train0", feature2_mean_train0)
    print("feature2_var_train0", feature2_var_train0)
    print("feature1_mean_train1", feature1_mean_train1)
    print("feature1_var_train1", feature1_var_train1)
    print("feature2_mean_train1", feature2_mean_train1)
    print("feature2_var_train1", feature2_var_train1)

#     print("feature1_mean_test0", feature1_mean_test0)
#     print("feature1_var_test0", feature1_var_test0)
#     print("feature2_mean_test0", feature2_mean_test0)
#     print("feature2_var_test0", feature2_var_test0)
#     print("feature1_mean_test1", feature1_mean_test1)
#     print("feature1_var_test1", feature1_var_test1)
#     print("feature2_mean_test1", feature2_mean_test1)
#     print("feature2_var_test1", feature2_var_test1)
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
    # print("test0_prediction: ", test0_prediction, type(test0_prediction))

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
    # print("test1_prediction: ", test1_prediction, type(test1_prediction))
    # End of task 3

    # Start of task 4
    print("\n\n")
    test0_accuracy = numpy.where(test0_prediction == True)
    test0_accuracy_perc = len(test0_accuracy[0]) / 980
    # print("test0_accuracy: ", test0_accuracy, type(test0_accuracy))
    print("test0_accuracy_perc: ", len(test0_accuracy[0]), test0_accuracy_perc, type(test0_accuracy_perc))

    test1_accuracy = numpy.where(test1_prediction == False)
    test1_accuracy_perc = len(test1_accuracy[0]) / 1135
    # print("test1_accuracy: ", test1_accuracy, type(test1_accuracy))
    print("test1_accuracy_perc: ", len(test1_accuracy[0]), test1_accuracy_perc, type(test1_accuracy_perc))
    # End of Task 4
     
    print("Final result in an array")
    final_array = [feature1_mean_train0, feature1_var_train0, feature2_mean_train0, feature2_var_train0,
                   feature1_mean_train1, feature1_var_train1, feature2_mean_train1, feature2_var_train1,
                   test0_accuracy_perc, test1_accuracy_perc]
    print(final_array)
    

if __name__ == '__main__':
    main()

