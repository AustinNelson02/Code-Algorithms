import numpy as np

class Kmeans:
    def __init__(self,k,tol, maxIter):
        self.k = k
        self.tol = tol
        self.maxIter = maxIter

    def fit(self, data):
        # Creating needed variables for later
        k = self.k
        tol = self.tol
        maxcount = self.maxIter
        m, n = data.shape

        #Permuting the indices of the amount of sets
        Ind = np.random.permutation(m)

        #Creating a group of lists for the indices
        self.Clusters_Ind = np.array_split(Ind,k)
        #Creating a dictionary that will hold the data
        self.Clusters = {}

        for i in range(k):
            self.Clusters_Ind[i] = list(self.Clusters_Ind[i])
            self.Clusters[i] = data[self.Clusters_Ind[i],:]

        #Creating a dictionary that will hold the centroids
        self.Centroids = {}

        #Computing the centroids of the data clusters
        for i in range(k):
            c = np.zeros(n)
            for j in range(len(self.Clusters[i])):
                c = c + self.Clusters[i][j]
            self.Centroids[i] = c/len(self.Clusters[i])


        counter = 0
        deltaQ = 500000
        Q = 0

        # Beginning the loop of the algorithm
        while deltaQ > tol and counter < maxcount:

            #Finding Q
            for i in range(k):
                for j in self.Clusters_Ind[i]:
                    cxDist = data[j,:] - self.Centroids[i]
                    Q = Q + np.dot(cxDist,cxDist)

            #Checking for any points that should change clusters
            for i in range(k):
                for j in self.Clusters_Ind[i]:
                    dist = []
                    for p in range(k):
                        cxdist = data[j,:] - self.Centroids[p]
                        dist.append(np.dot(cxdist,cxdist))
                    if dist[i] != min(dist):
                        self.Clusters_Ind[dist.index(min(dist))].append(j)
                        self.Clusters_Ind[i].pop(self.Clusters_Ind[i].index(j))

            #Updating Data Clusters
            for i in range(k):
                self.Clusters[i] = data[self.Clusters_Ind[i],:]

            #Updating Centroids
            for i in range(k):
                c = np.zeros(n)
                for j in range(len(self.Clusters[i])):
                    c = c + self.Clusters[i][j]
                self.Centroids[i] = c/len(self.Clusters[i])

            #Finding updated Q
            QAfter = 0
            for i in range(k):
                for j in self.Clusters_Ind[i]:
                    cxDist = data[j,:] - self.Centroids[i]
                    QAfter = QAfter + np.dot(cxDist,cxDist)

            #Finding deltaQ
            deltaQ = np.abs(Q - QAfter)
            Q = QAfter

            counter = counter + 1
        #Print out the average distance of data points to centroids
        print(Q/m)


        

Xtrain = np.genfromtxt('C:\\Users\\nelso\\Python\\zip.train', delimiter=' ')
XTest = np.genfromtxt('C:\\Users\\nelso\\Python\\zip.test', delimiter=' ')
m, n = Xtrain.shape
imagelabels = Xtrain[:,0]
labels = np.array(range(0,10,1))
J = np.argwhere(imagelabels <4)
Xtrain = Xtrain[J[:,0],:]

kmeans = Kmeans(5,.05,100)
kmeans.fit(Xtrain)

            


