from Precode import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')

k1,i_point1,k2,i_point2 = initial_S1('5569') # please replace 0111 with your last four digit of your ID

print(k1)
print(i_point1)
print(k2)
print(i_point2)
print(i_point1[0], i_point1[0][0], i_point1[0][1])


def get_data_as_df(data):
    data_df = pd.DataFrame(data, columns=['x', 'y'])
    return data_df


def create_centroids(k_value, i_points):
    centroid_array = []
    for index in range(k_value):
        print(index)
        centroid_array.append(i_points[index])

    return centroid_array

#############################################################
################## Using Numpy Start ########################
#############################################################
def get_distance_from_centroid(data, centroid_array):
    centroid_distance = []
    centroid_assignment = []
    # centroid_assignment = {}
    for index in range(len(data)):
        distance_dict = {}
        for centroid_index in range(len(centroid_array)):
            distance_dict[centroid_index] = np.sqrt(
                np.square(data[index][0] - centroid_array[centroid_index][0]) + np.square(
                    data[index][1] - centroid_array[centroid_index][1]))

        print(index, distance_dict, min(distance_dict, key=distance_dict.get))
        centroid_distance.append(distance_dict)
        centroid_assignment.append(min(distance_dict, key=distance_dict.get))

        '''
        assignment_cluster = min(distance_dict, key=distance_dict.get)
        if assignment_cluster in centroid_assignment.keys():
            # print(centroid_assignment[assignment_cluster], type(centroid_assignment[assignment_cluster]))
            # print(np.append(centroid_assignment[assignment_cluster], data[index]))
            existing = centroid_assignment[assignment_cluster]
            print(existing)
            centroid_assignment[assignment_cluster] = existing.append(data[index])
            # np.append(centroid_assignment[assignment_cluster], data[index])
        else:
            centroid_assignment[assignment_cluster] = [].append(data[index])
            #np.array(data[index])
        '''
    return centroid_distance, centroid_assignment


def recalculate_centroids(data, centroid_array, centroid_assignment):
    new_centroid_array = []
    for centroid_index in range(len(centroid_array)):
        x_sum = 0
        y_sum = 0
        total = 0
        for assignment_index in range(len(centroid_assignment)):
            if centroid_assignment[assignment_index] == centroid_index:
                x_sum += data[assignment_index][0]
                y_sum += data[assignment_index][1]
                total += 1

        x_value = x_sum / total
        y_value = y_sum / total

        new_centroid_array.append([x_value, y_value])

    return new_centroid_array


def loss_function(data, centroid_array):
    total = 0
    for centroid_index in range(len(centroid_array)):
        for assignment_index in range(len(centroid_assignment)):
            if centroid_assignment[assignment_index] == centroid_index:
                total += np.square(data[assignment_index][0] - centroid_array[centroid_index][0]) + \
                         np.square(data[assignment_index][1] - centroid_array[centroid_index][1])

    return total


def printGraph(data, centroids, annotate=""):
    #fig = plt.figure(figsize=(5,5))
    x = [lists[0] for lists in data]
    y = [lists[1] for lists in data]
    plt.rcParams['figure.figsize'] = (16, 9)
    # plt.style.use('ggplot')
    plt.scatter(x, y, color='k')
    colmap = {1: 'r', 2: 'y', 3: 'b', 4: 'c', 5: 'm'}
    for i in range(len(centroids)):
        if annotate != "":
            plt.scatter(*centroids[i], marker='*', s=200, c='w')
        else:
            plt.scatter(*centroids[i], marker='*', s=200, c=colmap[i + 1])


#############################################################
##################### Using Numpy End #######################
#############################################################


#############################################################
################## Using Pandas Start #######################
#############################################################
def get_distance_from_centroid_df(data_df, centroid_array):
    for centroid_index in range(len(centroid_array)):
        data_df['distance_from_{}'.format(centroid_index)] = np.sqrt(
            np.square(data_df['x'] - centroid_array[centroid_index][0]) +
            np.square(data_df['y'] - centroid_array[centroid_index][1]))
    centroids_cols = ['distance_from_{}'.format(i) for i in range(len(centroid_array))]
    data_df['nearest'] = data_df.loc[:, centroids_cols].idxmin(axis=1).apply(lambda x: int(x.lstrip('distance_from_')))
    return data_df


def recalculate_centroids_df(data_df, centroid_array):
    for centroid_index in range(len(centroid_array)):
        centroid_array[centroid_index][0] = np.mean(data_df[data_df['nearest'] == centroid_index]['x'])
        centroid_array[centroid_index][1] = np.mean(data_df[data_df['nearest'] == centroid_index]['y'])
    return centroid_array


def loss_function_df(data_df, centroid_array):
    totalCost = 0
    for centroid_index in range(len(centroid_array)):
        data = data_df[data_df['nearest'] == centroid_index]
        totalCost = totalCost + np.sum(
            (np.square(data['x'] - centroid_array[centroid_index][0]) +
             np.square(data['y'] - centroid_array[centroid_index][1])))
    return totalCost


def printGraph_df(df, centroids, annotate=""):
    #fig = plt.figure(figsize=(5,5))
    plt.rcParams['figure.figsize'] = (16, 9)
    # plt.style.use('ggplot')
    plt.scatter(df['x'], df['y'], color='k')
    colmap = {1: 'r', 2: 'y', 3: 'b', 4: 'c', 5: 'm'}
    for i in range(len(centroids)):
        if annotate != "":
            plt.scatter(*centroids[i], marker='*', s=200, c='w')
        else:
            plt.scatter(*centroids[i], marker='*', s=200, c=colmap[i + 1])

#############################################################
##################### Using Pandas End ######################
#############################################################

if __name__ == '__main__':
    data_df = get_data_as_df(data)

    ##################### Using Numpy Start ######################
    # for K1
    # Initiate
    centroid_array = create_centroids(k1, i_point1)

    centroid_distance, centroid_assignment = get_distance_from_centroid(data, centroid_array)
    # print(centroid_distance)
    # print(centroid_assignment)
    new_centroid_array = recalculate_centroids(data, centroid_array, centroid_assignment)
    printGraph(data, new_centroid_array, annotate="")
    # print(new_centroid_array)
    while True:
        previous_centroid = new_centroid_array
        centroid_distance, centroid_assignment = get_distance_from_centroid(data, new_centroid_array)
        # print(centroid_distance)
        # print(centroid_assignment)
        new_centroid_array = recalculate_centroids(data, new_centroid_array, centroid_assignment)
        # printGraph(data, new_centroid_array, annotate="")

        if previous_centroid.__eq__(new_centroid_array):
            break

    print("Final centroid for k1: ", new_centroid_array)
    printGraph(data, new_centroid_array, annotate="")
    total_loss = loss_function(data, new_centroid_array)
    print("Total Loss for k1: ", total_loss)

    # for K2
    # Initiate
    centroid_array = create_centroids(k2, i_point2)

    centroid_distance, centroid_assignment = get_distance_from_centroid(data, centroid_array)
    # print(centroid_distance)
    # print(centroid_assignment)
    new_centroid_array = recalculate_centroids(data, centroid_array, centroid_assignment)
    printGraph(data, new_centroid_array, annotate="")
    # print(new_centroid_array)
    while True:
        previous_centroid = new_centroid_array
        centroid_distance, centroid_assignment = get_distance_from_centroid(data, new_centroid_array)
        # print(centroid_distance)
        # print(centroid_assignment)
        new_centroid_array = recalculate_centroids(data, new_centroid_array, centroid_assignment)
        # printGraph(data, new_centroid_array, annotate="")

        if previous_centroid.__eq__(new_centroid_array):
            break

    print("Final centroid for k2: ", new_centroid_array)
    printGraph(data, new_centroid_array, annotate="")
    total_loss = loss_function(data, new_centroid_array)
    print("Total Loss for k2: ", total_loss)
    ##################### Using Numpy End ######################

    ##################### Using Pandas Start ######################
    # for K1
    # Initiate
    centroid_array = create_centroids(k1, i_point1)

    data_df = get_distance_from_centroid_df(data_df, centroid_array)
    # print(centroid_distance)
    # print(centroid_assignment)
    new_centroid_array = recalculate_centroids_df(data_df, centroid_array)
    printGraph_df(data_df, new_centroid_array, annotate="")
    # print(new_centroid_array)
    while True:
        previous_assignment = data_df['nearest'].copy(deep=True)
        data_df = get_distance_from_centroid_df(data_df, new_centroid_array)
        # print(centroid_distance)
        # print(centroid_assignment)
        new_centroid_array = recalculate_centroids_df(data_df, new_centroid_array)
        # printGraph(data_df, new_centroid_array, annotate="")

        if previous_assignment.equals(data_df['nearest']):
            break

    print("Final centroid for k1: ", new_centroid_array)
    printGraph_df(data_df, new_centroid_array, annotate="")
    total_loss = loss_function_df(data_df, new_centroid_array)
    print("Total Loss for k1: ", total_loss)

    # for K2
    # Initiate
    centroid_array = create_centroids(k2, i_point2)

    data_df = get_distance_from_centroid_df(data_df, centroid_array)
    # print(centroid_distance)
    # print(centroid_assignment)
    new_centroid_array = recalculate_centroids_df(data_df, centroid_array)
    printGraph_df(data_df, new_centroid_array, annotate="")
    # print(new_centroid_array)
    while True:
        previous_assignment = data_df['nearest'].copy(deep=True)
        data_df = get_distance_from_centroid_df(data_df, new_centroid_array)
        # print(centroid_distance)
        # print(centroid_assignment)
        new_centroid_array = recalculate_centroids_df(data_df, new_centroid_array)
        # printGraph_df(data_df, new_centroid_array, annotate="")

        if previous_assignment.equals(data_df['nearest']):
            break

    print("Final centroid for k2: ", new_centroid_array)
    printGraph_df(data_df, new_centroid_array, annotate="")
    total_loss = loss_function_df(data_df, new_centroid_array)
    print("Total Loss for k2: ", total_loss)

    ##################### Using Pandas End ######################