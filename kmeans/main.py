import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import kmeans
import argparse 

class Clustering:
    def __init__(self, ndata, dim):
        self.data = None
        self.dim = dim

        self.generate_data(ndata)
    
    def generate_data(self, ndata):
        self.data = np.random.random((ndata, self.dim))

    def plot_data(self):
        if self.dim == 2:
            fig = plt.figure(figsize=(8,4))

            ax1 = fig.add_subplot(121)
            ax1.scatter(self.data[:,0], self.data[:,1], s = 30)
            ax1.set_title('Data points')

            ax2 = fig.add_subplot(122)
            ax2.scatter(self.data[:,0], self.data[:,1], c = self.predictions, s = 30)
            ax2.scatter(self.centers[:,0], self.centers[:,1], s = 50, c = 'red', marker = '+', label='Centroids')
            ax2.legend()
            ax2.set_title('Clustered Data points')

            fig.suptitle("Kmeans Clustering")
            plt.savefig('assets/clusters.png')
            plt.show()
           
        else:
            # Creating figure
            fig = plt.figure()
            ax = plt.axes(projection = "3d")
            ax.scatter3D(self.data[:,0], self.data[:,1], self.data[:,2], c = self.predictions, s = 50)
            ax.scatter(self.centers[:,0], self.centers[:,1],  self.centers[:,2], s = 50, c = 'red', marker = '+', label='Centroids')
            ax.legend()
            plt.title("Clustered data points")
            plt.savefig('assets/clusters3d.png')
            plt.show()

    def main(self, ncluster):
        km = kmeans.KMeans(n_centroids = ncluster)
        km.fit(self.data)
        self.predictions = km.predict(self.data)
        self.centers = km.centroids

        self.plot_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('KMeans Clustering for 2D/3D data.')
    parser.add_argument('--dim', type = int, default = 2, help='Dimension of data to be generated.')
    parser.add_argument('--ncluster', type = int, default = 5, help='Number of clusters for cluster.')
    parser.add_argument('--ndata', type = int, default = 100, help='Number of data points for clustering.')
    args = parser.parse_args()

    Clustering(args.ndata, args.dim).main(args.ncluster)
