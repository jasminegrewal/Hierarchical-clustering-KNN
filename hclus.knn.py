import csv
import math
import operator
import numpy as np
import scipy
from operator import add
from scipy.spatial import distance

'''This function is called first with a filename containing training data and method which is taken from user
(single, average or complete) 
''' 
def hclus(filename, method):
    data=dict()
    trainee_vectors=[]
    indexes=[]
    '''reading the file, storing the indexes of data in indexes list and corresponding data in trainee_vectors for 
    later use'''
    with open(filename, 'r') as csvfile:
        indata = csv.reader(csvfile, delimiter=',', quotechar='|')
        p=0;
        for row in indata:
                indexes.append(p)
                data[row[0]]=[float(x) for x in row[0:]]
                trainee_vectors.append(data[row[0]])
                p=p+1;
                    
    '''Distance matrix "DM" is calculated using scipy's(python library) spatial distance'''
    DM=scipy.spatial.distance.cdist(trainee_vectors,trainee_vectors, metric='euclidean', p=2,V=None, VI=None, w=None)
    '''This gives the row and columns of the matrix DM which is square matrix in this case'''
    row,col=DM.shape
    '''find Min distance from all tuples'''
    def mindist(DM):
        row,col=DM.shape
        iu1 = np.triu_indices(row,1)
        return np.where(DM==np.min(DM[iu1]))
        
    '''len_mat is used to keep record of the increasing cluster number, means it is increased by one everytime
    a new cluster is formed and linkage is used to store the which two clusters are joined at a point along
    with the distance between them'''    
    
    len_mat=row
    linkage=[]
    label_list=np.array([x for x in range(row)])      

    '''clusters is a dictionary in which key is the index of data and value is the list with features 
    at that data point'''    
    '''cluster_index is a list of lists which is used to store which clusters are joined together, that is,
    it starts with singleton clusters' and in the end contains indexes in final clusters formed'''    

    clusters = dict(zip(indexes, trainee_vectors))    
    cluster_index=[]
    for i in range(len_mat):
        cluster_index.append([i])    # initially,every point is added as one cluster

    ''' modifying the distance matrix D at each point depending on the method used. Here this while condition
    is used to stop the formation of clusters at a desired point, In this case, five clusters will be formed'''
    
    k=int(raw_input("Enter the number of clusters \n")) 
    while(DM.shape!=(k,k)):
    	tuple_max,tuple_2=mindist(DM) # distance at which clusters are merged and it will at two places in DM (in upper and lower triangle)
        val1=tuple_max[0]             # always smaller
        val2=tuple_max[1]
        d_max_value=DM[val1][val2]     #distance value at that point
        DM=np.array(DM)
        
        v=cluster_index[val2]
        '''data index where cluster is formed are appended with eachother in cluster_index'''
        cluster_index.remove(v)  

        x=0
        for i in v:
            x=i
            cluster_index[val1].append(x)
                   
        '''merged clusters and distance between them is stored in linkag matrix'''    
        linkage.append([label_list[val1],label_list[val2],d_max_value])
        label_list=np.delete(label_list,val2)  #deleting label with val2
        label_list[val1]=len_mat               #creating next cluster label 
        len_mat+=1                             #increasing next cluster number by one

        if(method=="single"):
            n=np.minimum(DM[val1],DM[val2])     #in single method, using minimum distance to join clusters
        elif(method=="complete"):
            n=np.maximum(DM[val1],DM[val2])     #in complete, maximum distance is used
        elif(method=="average"):
            r,c=DM.shape                       #in average, average distance is used
            arr=[]
            for i in range(r):
                arr.append(np.sum(DM[i])/r)
            n=np.array(arr)
      
        '''deleting old values from distance matrix DM and adding new ones' depending on the method used above'''
        val=DM[0,1]
        new_mat=np.delete(n,val2,0)
        DM=np.delete(DM,val2,0)              #deleting row
        DM=np.delete(DM,val2,1)              #DELETING COLUMN
        DM[val1]=new_mat                    #replacing old row with new in case valuea are different
        DM[:,val1]=new_mat                  #inserting column
        DM[val1,val1]=0                     
        if(DM.shape==(1,1)):
             DM[val1,val1]=val

    return(linkage,DM,clusters,cluster_index)
    
#getting method input from user
mthd=str(raw_input("Enter method to be used for clustering i.e single, complete or average\n") )          

#calling hierarchicaal clustering function to perform clustering
Z,DM,clusters,cluster_index =hclus('train_seeddata.csv',mthd)

#this prints the indexes of data points merged into each cluster
print("\n cluster_index \n")
print '\n'.join(map(str,cluster_index))

#this prints the linkage matrix with cluster numbers and distance between them
print("\n Linkage matrix \n")
print '\n'.join(map(str,Z))

#the resulting distance matrix between the final clusters formed
print("\n Distance Matrix \n")
print '\n'.join(map(str,DM))

'''Final is a dictionary in which key is the cluster ID and value is the list containg the average of features 
which are included in that cluster'''
final=dict()
m=len(cluster_index)
#print m
for i in range(m):
    count=0;
    for lists in cluster_index[i]:
        if count==0:
            final[i]=clusters[i]
            count+=1
        elif count>0:
            final[i]=map(add,final[i],clusters[lists])
            count+=1
    final[i]=[x/count for x in final[i]]
print("\nCluster IDs and average point for a cluster")
for x in final:
    print ("\nCluster Id  " +str(x))
    print (final[x])

'''Now, on the test data nearest neighbor is implemented, first knn is called with file containg test data'''
def knn(filename):
    with open(filename, 'r') as csvfile:
        d=dict()
        '''test data file is read and data is stored in testlists'''
        testlists=[]
        testdata = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in testdata:                
                d[row[0]]=[float(x) for x in row[0:]]
                testlists.append(d[row[0]])
    
    resultmatrix =dict()                   #dictionary to store end results
     
    '''neighbors method is called for all test data lists stored in testlists and clus will contain the cluster 
    Id to which test data belongs with a minimum distance '''   
    for k in range(len(testlists)):
        clus=neighbors(final,testlists[k])
        
        resultmatrix[k]=clus   #resultmatrix is a dictionary where key is the test data index and 
                               #value has cluster id with minimum distance
    return resultmatrix
        
#calulates distance between set1 and set2 which are two lists of features in this case
def calcdistance(set1, set2, length):
	distance = 0
	for x in range(length):
		distance += pow((set1[x] - set2[x]), 2)
		#length tells till which point in the set1 and set2 array are to be used to calculate distance
	return math.sqrt(distance)    

def neighbors(givendata, sample):
	distances = []                 #to store distance of each data point in training data from sample point
	length = len(sample)
	for x in range(len(givendata)):
		dist = calcdistance(sample, givendata[x], length)
		distances.append((x, dist))
      #sort the distances to find minimum distance
	distances.sort(key=operator.itemgetter(1)) 
      #return the first distance as it is the minimum
	return(distances[0])

    
  
result_matrix = knn('seed_testdata.csv')

for x in result_matrix:
    print("\ntest_data number "+str(x)+ " belongs to cluster_ID " +str(result_matrix[x][0])+ " with distance " +str(result_matrix[x][1]))
   
    
