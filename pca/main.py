import numpy as np
import pandas as pd
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from pca import PrincipalComponentAnalysis

def read_data(file_path):
    """
    Read .csv, .txt file to create dataframe.
    """
    dataframe = pd.read_csv(file_path)
    
    return  dataframe

def visualize_data(result):
    """
    Visualize principal components.
    """
    viz_pc1 = plt
    viz_pc1.figure(figsize=(20, 10))
    sns.scatterplot(data = result, x = 'PC1', y = [0] * len(result), hue = "labels", s = 200)
    viz_pc1.savefig('assets/principal_component1.png')
    viz_pc1.show()

    viz_pc= plt
    viz_pc.figure(figsize=(20, 10))
    sns.scatterplot(data = result, x = 'PC1', y = 'PC2', hue = "labels", s = 200)
    viz_pc.savefig('assets/principal_component1.png')
    viz_pc.show()

def main(df, num_pc):
    Y_df = df[df.columns[-1]]
    df_X = df.iloc[:, :-1]
    PCA = PrincipalComponentAnalysis(num_pc)
    X_norm = PCA.normalize_data(df_X)
    Cov_X = PCA.covariance_matrix(X_norm)
    PCA.svd(Cov_X)

    principal_components = PCA.compress(X_norm)

    result = pd.DataFrame(principal_components[0], columns=['PC1'])
    for i in range(2, num_pc + 1):
        result['PC' + str(i)] = principal_components[i - 1]
    result['labels'] = Y_df

    visualize_data(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Principal Component Analysis')
    parser.add_argument('--num_pc', type = int, default = 2, help='Number of principal components.')
    parser.add_argument('--data_path', type = str, default = 'datasets/iris.csv', help='Input data (dataframe preferred).')
    args = parser.parse_args()

    df = read_data(args.data_path)

    main(df, args.num_pc)
    