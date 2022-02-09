Student ID : 1220707254, Week = 5, Project -2, Strategy-2

====================
Results
====================
centeriod1 = [[6.7953243228207842, 2.7877851195756893],[6.9282228481889643, 7.9218715211409867],[3.1966934324401133, 6.8712607992868691],[2.8523514931105352, 2.2818648297203241]]
cost1 = 803.2167238057567

centeriod2 = [[5.2305366674047375, 4.2793424960490993],[7.9143099778183128, 8.5199098077000759],[2.6819863341889287, 2.0946158678008095],[5.2402829638043125, 7.5313102932678238],[7.5561678223977253, 2.2351679598575336],[2.5416525233103435, 7.0026783235364887]]
cost2 = 462.92635582483734


========================Implementation for both centroid1 and centroid2 ===================
===========================================================================================


from Precode2 import *
import numpy
data = np.load('AllSamples.npy')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pl
print(pd.__version__)


k1,i_point1,k2,i_point2 = initial_S2('7254') # please replace 0111 with your last four digit of your ID


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# Code for K1 centroid

centroids ={1: [ 4.32239695 , 0.33088885]}
df = pd.DataFrame(data, columns=['x','y'])
for i in range(k1):
    df['distance_from_{}'.format(i+1)]=0.0
for i in range(k1-1):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt(np.square(df['x']-centroids[i][0]) + np.square(df['y']-centroids[i][1]))
    centroids_cols = ['distance_from_{}'.format(i+1) for i in range(k1)]
    #print(centroids_cols)
    df['mean_distance'] = df.loc[:,centroids_cols].sum(axis=1)/len(centroids)
    df['centroid_length'] = len(centroids)
    #print(df.head().to_string())
    #df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    #d = df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    #d.values.flatten()
    df_max = df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    centroids[len(centroids)+1] = df_max.values.flatten().tolist()
    df.drop(df_max.index.values, inplace=True)
print(centroids)

centroids
print("initial centroids : {}".format(centroids))
print(len(data))
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
    #colmap = {1:'r',2:'y',3:'b',4:'g',5:'g'}
    n = 20
    colmap = pl.cm.jet(np.linspace(0,1,n))
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


centroids ={1: [ 4.43990951 , 3.70495907]}
df = pd.DataFrame(data, columns=['x','y'])
for i in range(k2):
    df['distance_from_{}'.format(i+1)]=0.0
for i in range(k2-1):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt(np.square(df['x']-centroids[i][0]) + np.square(df['y']-centroids[i][1]))
    centroids_cols = ['distance_from_{}'.format(i+1) for i in range(k2)]
    #print(centroids_cols)
    df['mean_distance'] = df.loc[:,centroids_cols].sum(axis=1)/len(centroids)
    df['centroid_length'] = len(centroids)
    #print(df.head().to_string())
    #df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    #d = df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    #d.values.flatten()
    df_max = df[df['mean_distance'] == df['mean_distance'].max()][['x','y']]
    centroids[len(centroids)+1] = df_max.values.flatten().tolist()
    df.drop(df_max.index.values, inplace=True)
print(centroids)

centroids
print("initial centroids : {}".format(centroids))
print(len(data))
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
    #colmap = {1:'r',2:'y',3:'b',4:'g',5:'g'}
    n = 20
    colmap = pl.cm.jet(np.linspace(0,1,n))
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
print("Cost Function: {}".format(costFunction(df,centroids)))