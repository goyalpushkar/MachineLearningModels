import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='7254'
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
    
    ##Step -1 Started
    ##Converting training dataset to 5000x784
    train0T = train0.reshape(5000,784)
    train1T = train1.reshape(5000,784)
    
    ##Converting test dataset to 980x784 and 1135x784
    test0T = test0.reshape(980,784)
    test1T = test1.reshape(1135,784)
    
    ##feature extraction for image 0 (mean, standard deviation) in training dataset
    feature1_train0 = numpy.array([numpy.mean(image0) for image0 in train0T])
    feature2_train0 = numpy.array([numpy.std(image0) for image0 in train0T])

    ##feature extraction for image 1 (mean, standard deviation) in training dataset
    feature1_train1 = numpy.array([numpy.mean(image1) for image1 in train1T])
    feature2_train1 = numpy.array([numpy.std(image1) for image1 in train1T])
    
    
    ##feature extraction for image 0 (mean, standard deviation) in test dataset
    feature1_test0 = numpy.array([numpy.mean(image0) for image0 in test0T])
    feature2_test0 = numpy.array([numpy.std(image0) for image0 in test0T])
    
    ##feature extraction for image 1 (mean, standard deviation) in test dataset
    feature1_test1 = numpy.array([numpy.mean(image1) for image1 in test1T])
    feature2_test1 = numpy.array([numpy.std(image1) for image1 in test1T])
    
    ##Step - 1 Ended
    
    ##Step - 2 Started
    mean_feature1_train0 = numpy.mean(feature1_train0)
    var_feature1_train0 = numpy.var(feature1_train0)
    std_feature1_train0 = numpy.std(feature1_train0)
    
    mean_feature2_train0 = numpy.mean(feature2_train0)
    var_feature2_train0 = numpy.var(feature2_train0)
    std_feature2_train0 = numpy.std(feature2_train0)
    
    mean_feature1_train1 = numpy.mean(feature1_train1)
    var_feature1_train1 = numpy.var(feature1_train1)
    std_feature1_train1 = numpy.std(feature1_train1)
    
    mean_feature2_train1 = numpy.mean(feature2_train1)
    var_feature2_train1 = numpy.var(feature2_train1)
    std_feature2_train1 = numpy.std(feature2_train1)
    
    
    print("mean_feature1_train0: " + str(mean_feature1_train0))
    print("var_feature1_train0: " + str(var_feature1_train0))
    print(std_feature1_train0)
    print("mean_feature2_train0: " + str(mean_feature2_train0))
    print("var_feature2_train0: " + str(var_feature2_train0))
    print(std_feature2_train0)
    print("mean_feature1_train1: " + str(mean_feature1_train1))
    print("var_feature1_train1: " + str(var_feature1_train1))
    print(std_feature1_train1)
    print("mean_feature2_train1: " + str(mean_feature2_train1))
    print("var_feature2_train1: " + str(var_feature2_train1))
    print(std_feature2_train1)
    
    ##Step-2 ended
    
    
    ##Finding probability for image 0 test dataset
    ## Find a probability P(Y=1|X) => P(X|Y=1) * P(Y=1) / P(X) =>  P(X|Y=1) => P(X1|Y = 1) * P(X2 | Y=1)
    #PofX1givenYequal1_0= numpy.array([(1/std_feature1_train1*numpy.sqrt(2*numpy.pi))*numpy.exp(-numpy.square(feature1-mean_feature1_train1)/2*var_feature1_train1) for feature1 in feature1_test0])
    PofX1givenYequal1_0 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature1_train1)) * numpy.exp(-((feature1-mean_feature1_train1)**2 / (2 * var_feature1_train1 ))) for feature1 in feature1_test0])
    
    #PofX2givenYequal1_0 = numpy.array([(1/std_feature2_train1*numpy.sqrt(2*numpy.pi))*numpy.exp(-numpy.square(feature2-mean_feature2_train1)/2*var_feature2_train1) for feature2 in feature2_test0])
    PofX2givenYequal1_0 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature2_train1)) * numpy.exp(-((feature2-mean_feature2_train1)**2 / (2 * var_feature2_train1 ))) for feature2 in feature2_test0])
    
    PofYequal1_0 = PofXgivenYequal1_test1_0 = numpy.multiply(PofX1givenYequal1_0 , PofX2givenYequal1_0) 
    
    ##Find a probability P(Y=0|X) => P(X|Y=0) * P(Y=0) / P(X) => P(X|Y=0) => P(X1|y=0) * P(X2|y=0)
    #PofX1givenYequal0_0= numpy.array([(1/std_feature1_train0*numpy.sqrt(2*numpy.pi))*numpy.exp(-numpy.square(feature1-mean_feature1_train0)/2*var_feature1_train0) for feature1 in feature1_test0])
    PofX1givenYequal0_0 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature1_train0)) * numpy.exp(-((feature1-mean_feature1_train0)**2 / (2 * var_feature1_train0 ))) for feature1 in feature1_test0])
    
    #PofX2givenYequal0_0= numpy.array([(1/std_feature2_train0*numpy.sqrt(2*numpy.pi))*numpy.exp(-numpy.square(feature2-mean_feature2_train0)/2*var_feature2_train0) for feature2 in feature2_test0])
    PofX2givenYequal0_0 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature2_train0)) * numpy.exp(-((feature2-mean_feature2_train0)**2 / (2 * var_feature2_train0 ))) for feature2 in feature2_test0])
    
    PofYequal0_0 = PofXgivenYequal0_test1_0 = numpy.multiply(PofX1givenYequal0_0 , PofX2givenYequal0_0)
    
    test0_feature_predictions = (PofXgivenYequal1_test1_0 > PofXgivenYequal0_test1_0)
    #print(test0_feature_predictions)
    #print(type(test1_feature_predictions))
    Test0_accuracy = numpy.where(test0_feature_predictions == False)
    print(len(Test0_accuracy[0]))
    print(len(Test0_accuracy[0])/980)
    
    
    ### END for image 0
    
    
    
    ##Finding probability for image 1 test dataset
    ## Find a probability P(Y=1|X) => P(X|Y=1) * P(Y=1) / P(X) =>  P(X|Y=1) => P(X1|Y = 1) * P(X2 | Y=1)
    
    #PofX1givenYequal1_1= numpy.array([1/std_feature1_train1*numpy.sqrt(2*numpy.pi)*numpy.exp(-numpy.square(feature1-mean_feature1_train1)/2*var_feature1_train1) for feature1 in feature1_test1])
    #exponent_PofX1givenYequal1_1 = numpy.exp(-((feature1-mean_feature1_train1)**2 / (2 * var_feature1_train1 )))
    PofX1givenYequal1_1 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature1_train1)) * numpy.exp(-((feature1-mean_feature1_train1)**2 / (2 * var_feature1_train1 ))) for feature1 in feature1_test1])
    
    #PofX2givenYequal1_1 = numpy.array([1/std_feature2_train1*numpy.sqrt(2*numpy.pi)*numpy.exp(-numpy.square(feature2-mean_feature2_train1)/2*var_feature2_train1) for feature2 in feature2_test1])
    #exponent_PofX2givenYequal1_1 = numpy.exp(-((feature2-mean_feature2_train1)**2 / (2 * var_feature2_train1 )))
    PofX2givenYequal1_1 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature2_train1)) * numpy.exp(-((feature2-mean_feature2_train1)**2 / (2 * var_feature2_train1 ))) for feature2 in feature2_test1])
    
    PofYequal1_1 = PofXgivenYequal1_test1_1 = numpy.multiply(PofX1givenYequal1_1 , PofX2givenYequal1_1) 
    
    ##Find a probability P(Y=0|X) => P(X|Y=0) * P(Y=0) / P(X) => P(X|Y=0) => P(X1|y=0) * P(X2|y=0)
    #PofX1givenYequal0_1= numpy.array([1/std_feature1_train0*numpy.sqrt(2*numpy.pi)*numpy.exp(-numpy.square(feature1-mean_feature1_train0)/2*var_feature1_train0) for feature1 in feature1_test1])
    #exponent_PofX1givenYequal0_1 = numpy.exp(-((feature1-mean_feature1_train0)**2 / (2 * var_feature1_train0 )))
    PofX1givenYequal0_1 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature1_train0)) * numpy.exp(-((feature1-mean_feature1_train0)**2 / (2 * var_feature1_train0 ))) for feature1 in feature1_test1])
    
    
    #PofX2givenYequal0_1= numpy.array([1/std_feature2_train0*numpy.sqrt(2*numpy.pi)*numpy.exp(-numpy.square(feature2-mean_feature2_train0)/2*var_feature2_train0) for feature2 in feature2_test1])
    #exponent_PofX2givenYequal0_1 = numpy.exp(-((feature2-mean_feature2_train0)**2 / (2 * var_feature2_train0 )))
    PofX2givenYequal0_1 =numpy.array([(1 / (numpy.sqrt(2 * numpy.pi) * std_feature2_train0)) * numpy.exp(-((feature2-mean_feature2_train0)**2 / (2 * var_feature2_train0 ))) for feature2 in feature2_test1])
    
    PofYequal0_1 = PofXgivenYequal0_test1_1 = numpy.multiply(PofX1givenYequal0_1 , PofX2givenYequal0_1) 
    
    test1_feature_predictions = (PofXgivenYequal1_test1_1 > PofXgivenYequal0_test1_1)
    #print(test1_feature_predictions)
    #print(test1_feature_predictions == 'True')
    #print(type(test1_feature_predictions))
    Test1_accuracy = numpy.where(test1_feature_predictions == True)
    print(Test1_accuracy[0].size)
    print(len(Test1_accuracy[0])/1135)
    
    
    ###End of Image 1
    
    
if __name__ == '__main__':
    main()