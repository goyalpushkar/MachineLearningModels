
Student ID : 1220707254, Week = 5, Project -2, Strategy-1

====================
Results
====================
centeriod1 = [[4.8309195843563568, 7.2995995867233274],[7.2397511895844486, 2.4820826910731952],[3.2348900463590846, 2.5530321964002027]]
cost1 = 1338.1076016520997

centeriod2 = [[3.2220235470535226, 7.1593799610031779],[7.4936536725150615, 8.5241795241103411],[5.3751437877396455, 4.5310165361189032],[2.6819863341889287, 2.0946158678008095],[7.5561678223977253, 2.2351679598575336]]
cost2 = 592.0694342732749



========================Implementation for both centroid1 and centroid2 ===================
===========================================================================================

from Precode import *
import numpy
data = np.load('AllSamples.npy')
import matplotlib.pyplot as plt
import pandas as pd


k1,i_point1,k2,i_point2 = initial_S1('7254') # please replace 0111 with your last four digit of your ID


print(k1)
print(i_point1)
print(k2)
print(i_point2)

# Code for K1 centroid

centroids = {
    i+1:[i_point1[i][0], i_point1[i][1]]
    #i+1:[np.random.uniform(0,10), np.random.uniform(0,10)]
    for i in range(k1)
}
print("initial centroids : {}".format(centroids))
df = pd.DataFrame(data, columns=['x','y'])
xypoint = np.array(df[['x','y']])
df['xypoint'] = xypoint.tolist()
def assignment(df,centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt(np.square(df['x']-centroids[i][0]) + np.square(df['y']-centroids[i][1]))
    centroids_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['nearest'] = df.loc[:,centroids_cols].idxmin(axis=1).apply(lambda x:int(x.lstrip('distance_from_')))
    return df

def updateCentroid(centroids):
    for i in centroids.keys():
        centroids[i][0] = numpy.mean(df[df['nearest'] == i]['x'])
        centroids[i][1] = numpy.mean(df[df['nearest'] == i]['y'])
    return centroids

def costFunction(df, centroids):
    totalCost=0
    for i in centroids.keys():
        data = df[df['nearest'] == i]
        totalCost = totalCost + np.sum((np.square(data['x']-centroids[i][0]) + np.square(data['y']-centroids[i][1])))
    return totalCost

def printGraph(df, centroids, annotate=""):
    #fig = plt.figure(figsize=(5,5))
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    plt.scatter(df['x'], df['y'], color='k')
    colmap = {1:'r',2:'y',3:'b'}
    for i in centroids.keys():
        if(annotate != ""):
            plt.scatter(*centroids[i],marker='*', s=200, c='w')
        else:
            plt.scatter(*centroids[i],marker='*', s=200, c=colmap[i])

assignment(df,centroids)
centroids =  updateCentroid(centroids)
printGraph(df,centroids,"Start")
while True:
    previous_assignment = df['nearest'].copy(deep=True)
    #print(previous_assignment)
    assignment(df,centroids)
    centroids =  updateCentroid(centroids)
    #print(centroids)
    print(centroids)
    printGraph(df,centroids)
    
    if(previous_assignment.equals(df['nearest'])):
        break
print("Final centroids : {}".format(centroids))
#df.head()
print("Cost Function: {}".format(costFunction(df,centroids)))


# Code for K2 centroid

centroids = {
    i+1:[i_point2[i][0], i_point2[i][1]]
    #i+1:[np.random.uniform(0,10), np.random.uniform(0,10)]
    for i in range(k2)
}
print("initial centroids : {}".format(centroids))
df = pd.DataFrame(data, columns=['x','y'])
xypoint = np.array(df[['x','y']])
df['xypoint'] = xypoint.tolist()
def assignment(df,centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt(np.square(df['x']-centroids[i][0]) + np.square(df['y']-centroids[i][1]))
    centroids_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['nearest'] = df.loc[:,centroids_cols].idxmin(axis=1).apply(lambda x:int(x.lstrip('distance_from_')))
    return df

def updateCentroid(centroids):
    for i in centroids.keys():
        centroids[i][0] = numpy.mean(df[df['nearest'] == i]['x'])
        centroids[i][1] = numpy.mean(df[df['nearest'] == i]['y'])
    return centroids

def costFunction(df, centroids):
    totalCost=0
    for i in centroids.keys():
        data = df[df['nearest'] == i]
        totalCost = totalCost + np.sum((np.square(data['x']-centroids[i][0]) + np.square(data['y']-centroids[i][1])))
    return totalCost

def printGraph(df, centroids, annotate=""):
    #fig = plt.figure(figsize=(5,5))
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    plt.scatter(df['x'], df['y'], color='k')
    colmap = {1:'r',2:'y',3:'b',4:'c',5:'m'}
    for i in centroids.keys():
        if(annotate != ""):
            plt.scatter(*centroids[i],marker='*', s=200, c='w')
        else:
            plt.scatter(*centroids[i],marker='*', s=200, c=colmap[i])

assignment(df,centroids)
centroids =  updateCentroid(centroids)
printGraph(df,centroids,"Start")
while True:
    previous_assignment = df['nearest'].copy(deep=True)
    #print(previous_assignment)
    assignment(df,centroids)
    centroids =  updateCentroid(centroids)
    #print(centroids)
    print(centroids)
    printGraph(df,centroids)
    
    if(previous_assignment.equals(df['nearest'])):
        break
print("Final centroids : {}".format(centroids))
#df.head()
print("Cost Function: {}".format(costFunction(df,centroids)))