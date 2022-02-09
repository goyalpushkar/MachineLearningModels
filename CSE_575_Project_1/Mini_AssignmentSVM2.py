'''
Given an input array of binary feature values for a single feature, f,
and an input array of binary class labels, y,
write a function that computes P(f=0|y=1) and P(f=1|y=1)
Your function should return the computed values in an array, in the form
[P(f=0|y=1), P(f=1|y=1)]
'''
# from __future__ import division
def likelihood_estimate(features, labels):
    count_y1 = 0
    count_y0 = 0
    for elem in labels:
        if elem == 1:
            count_y1 += 1
        else:
            count_y0 += 1
    # prob_y1 = count_y1 / len(labels)
    # prob_y0 = count_y0 / len(labels)
    print("count_y1: "+str(count_y1) + ", count_y0: "+str(count_y0))

    # P(f=0|y=1) = P(f=0, y=1)/ P(y=1)
    count_f0_y1 = 0
    for index in range(len(features)):
        if features[index] == 0 and labels[index] == 1:
            count_f0_y1 += 1

    prob_f0_y1 = (float(count_f0_y1) / float(count_y1))

    print("count_f0_y1: "+str(count_f0_y1) + ", prob_f0_y1: "+str(prob_f0_y1))
    # P(f=1|y=1) = P(f=1, y=1)/ P(y=1)
    count_f1_y1 = 0
    for index in range(len(features)):
        if features[index] == 1 and labels[index] == 1:
            count_f1_y1 += 1

    prob_f1_y1 = (float(count_f1_y1) / float(count_y1))
    print("count_f1_y1: " + str(count_f1_y1) + ", prob_f1_y1: " + str(prob_f1_y1))

    return_array = [prob_f0_y1, prob_f1_y1]

    return return_array


if __name__ == '__main__':
    # Test case 1
    # features = [0, 1, 1, 1, 0, 1, 0]
    # labels = [1, 1, 1, 1, 0, 0, 0]

    # Test case 2
    features = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    labels = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # Output = [0.49999999999, 0.49999999999]

    # Test case 3
    features = [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    output = [0.5833333333333333, 0.41666666666666663]
    final_value = likelihood_estimate(features, labels)

    print(final_value)