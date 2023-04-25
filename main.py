# This is a sample Python script.
import pandas as pd
import sklearn.mixture as mixture
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def import_sonar_data():
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',header=None)
        print(data)
        data = data.replace({'R': 0}, regex=True)
        data = data.replace({'M': 1}, regex=True)
        df2 = data.iloc[: , :-1]
        target = data.iloc[:,-1:]
        t_np = target.to_numpy()
        np = df2.to_numpy()
        t_np = t_np.flatten()
        return np,t_np
def import_synthetic_data(name):
    data = pd.read_csv(name, delimiter=' ')
    np = data.to_numpy()
    return np


def silhouette_score(data, predictions):
    return metrics.silhouette_score(X= data, labels = predictions, metric= 'euclidean')

def mutual_info(target, pred):
    return metrics.mutual_info_score(labels_true = target, labels_pred = pred)


def plot_model(model,data,mixtures,colors):
    #plot best model
    model.fit(data)
    predictions = model.predict(data)
    for j in range(len(mixtures)):
        for i in range(len(data)):
            if predictions[i] == j:
                plt.scatter(x=data[i][0],y=data[i][1],c=colors[j], s = 10)

    for u in model.means_:
        plt.scatter(u[0],u[1], c = 'black', s = 20)

    for i in range(len(mixtures)):
        u = model.means_[i][0]  # x-position of the center
        v = model.means_[i][1]  # y-position of the center

        cov = model.covariances_[i]

        eigvals, eigvecs = np.linalg.eigh(cov)
        max_idx = np.argmax(eigvals)
        a = np.sqrt(eigvals[max_idx])
        b = np.sqrt(eigvals[1 - max_idx])
        t = np.arctan2(eigvecs[max_idx, max_idx], eigvecs[1-max_idx, max_idx])

        midpoint = []
        midpoint.append(u)
        midpoint.append(v)

        ellipse = Ellipse(xy = midpoint, width = 4 * a, height= 4 * b,
                          angle =  np.degrees(t), edgecolor='black',facecolor= None,fill = False)
        plt.gca().add_patch(ellipse)

    plt.show()

def plot_syntetic_by_mixtures(Data, Range):
    #plots the likelyhood and siluete score for each cluster number used
    silhouette_scores = []
    likelihoods = []
    reg = 1e-6
    for i in Range:
        model = mixture.GaussianMixture(n_components= i, covariance_type='full',
                                    reg_covar=reg, max_iter= 200, init_params='k-means++')
        model.fit(Data)
        predictions = model.predict(Data)
        silhouette_scores.append(silhouette_score(Data,predictions))
        likelihoods.append(model.score(Data))

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of Mixtures')
    ax1.set_ylabel('likelihood score', color=color)
    ax1.plot(Range, likelihoods, color=color, label='likelihood score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Silhouette score', color=color)  # we already handled the x-label with ax1
    ax2.plot(Range, silhouette_scores, color=color, label='Silhouette score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title("GMM Preformace Based on Number of Mixtures")

    plt.show()

def plot_synthetic_by_iter(Data, Range,clusters):
    silhouette_scores = []
    likelihoods = []
    model = mixture.GaussianMixture(n_components= clusters, covariance_type='full',
                                    reg_covar=1e-6, max_iter= 0, init_params='kmeans')
    model.fit(Data)


    for i in Range:
        mean = model.means_
        pre = model.precisions_
        weights = model.weights_
        model = mixture.GaussianMixture(n_components= clusters, means_init= mean, precisions_init= pre,
                                    reg_covar=1e-6, max_iter=1, weights_init= weights,
                                    warm_start= True)
        model.fit(Data)
        predictions = model.predict(Data)
        silhouette_scores.append(silhouette_score(Data,predictions))
        likelihoods.append((model.score(Data)))

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('likelihood score', color=color)
    ax1.plot(Range, likelihoods, color=color, label = 'likelihood score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Silhouette score', color=color)  # we already handled the x-label with ax1
    ax2.plot(Range, silhouette_scores, color=color, label='Silhouette score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title("GMM Preformace Based on Number of iterations")

    plt.show()


def plot_sonar_by_iter(Data, Range, target_data,clusters):
    Mutual_Info = []
    likelihoods = []
    silhouette_scores = []
    model = mixture.GaussianMixture(n_components=clusters, covariance_type='full',
                                    reg_covar=1e-6, max_iter=0, init_params='kmeans')
    model.fit(Data)

    for i in Range:
        mean = model.means_
        pre = model.precisions_
        weights = model.weights_
        model = mixture.GaussianMixture(n_components=clusters, means_init=mean, precisions_init=pre,
                                        reg_covar=1e-6, max_iter=1, weights_init=weights,
                                        warm_start=True)
        model.fit(Data)
        predictions = model.predict(Data)
        Mutual_Info.append(mutual_info(target= target_data,pred= predictions))
        likelihoods.append((model.score(Data)))
        silhouette_scores.append(silhouette_score(Data, predictions))

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('likelihood score', color=color)
    ax1.plot(Range, likelihoods, color=color, label='likelihood score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Mutual Information', color=color)  # we already handled the x-label with ax1
    ax2.plot(Range, Mutual_Info, color=color, label='Mutual Information')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title("GMM Preformace Based on Number of iterations")

    plt.show()
    fig, ax2 = plt.subplots()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette score', color=color)  # we already handled the x-label with ax1
    ax2.plot(Range, silhouette_scores, color=color, label='Silhouette score')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("GMM Preformace Based on Number of iterations")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Gus Vietze, gvietze@u.rochester.edu")
    ds = input("what data set would you like to test: A (A), B (B), C (C), Z (Z), Sonar (S)")
    if ds == 'A':
        path = input("enter the path to the synthetic data")
        best_model = input("would you like to plot the best model (0), "
                       "the model performance based on iteration(1), "
                       "or the model performance based on mixture number(2)")
        if best_model == '0':
            plot_model(model = mixture._gaussian_mixture(n_components=6, covariance_type='full',
                                    reg_covar=1e-6, max_iter=100, init_params='kmeans'),
                       data = import_synthetic_data('A'),
                       mixtures= [0,1,2,3,4,5],
                       colors= ['red', 'yellow', 'green','blue', 'orange', 'purple'])
        elif best_model == '1':
            plot_synthetic_by_iter(Data = import_synthetic_data(path),Range = range(0,20),clusters=6)
        elif best_model == '2':
            plot_syntetic_by_mixtures(Data=import_synthetic_data(path), Range=range(2, 10))
        else : print('please enter a valid input')

    elif ds == 'B':
        path = input("enter the path to the synthetic data")
        best_model = input("would you like to plot the best model (0), "
                           "the model performance based on iteration(1), "
                           "or the model performance based on mixture number(2)")
        if best_model == '0':
            plot_model(model = mixture._gaussian_mixture(n_components=3, covariance_type='full',
                                    reg_covar=1e-6, max_iter=100, init_params='kmeans'),
                       data = import_synthetic_data('B'),
                       mixtures= [0,1,2],
                       colors= ['red', 'yellow','blue'])
        elif best_model == '1':
            plot_synthetic_by_iter(Data=import_synthetic_data(path), Range=range(0, 20), clusters=4)
        elif best_model == '2':
            plot_syntetic_by_mixtures(Data=import_synthetic_data(path), Range=range(2, 10))
        else:
            print('please enter a valid input')

    elif ds == 'C':
        path = input("enter the path to the synthetic data")
        best_model = input("would you like to plot the best model (0), "
                           "the model performance based on iteration(1), "
                           "or the model performance based on mixture number(2)")
        if best_model == '0':
            plot_model(model = mixture._gaussian_mixture(n_components=8, covariance_type='full',
                                    reg_covar=1e-6, max_iter=100, init_params='kmeans'),
                       data = import_synthetic_data('C'),
                       mixtures= [0,1,2,3,4,5,6,7],
                       colors= ['red', 'yellow', 'green','blue', 'orange', 'purple','brown','pink'])
        elif best_model == '1':
            plot_synthetic_by_iter(Data=import_synthetic_data(path), Range=range(0, 20), clusters=8)

        elif best_model == '2':
            plot_syntetic_by_mixtures(Data = import_synthetic_data(path),Range= range(2,10))
        else:
            print('please enter a valid input')
    elif ds == 'Z':
        path = input("enter the path to the synthetic data")
        best_model = input(" would you like to plot the model performance based on iteration(0), "
                           "or the model performance based on mixture number(1)")
        if best_model == '0':
            plot_synthetic_by_iter(Data=import_synthetic_data(path), Range=range(0, 20), clusters=14)

        elif best_model == '1':
            plot_syntetic_by_mixtures(Data = import_synthetic_data(path),Range= range(2,20))
        else:
            print('please enter a valid input')
    elif ds == 'S':
            data, target = import_sonar_data()
            plot_sonar_by_iter(Data = data , Range=range(0, 20), clusters=2, target_data= target)
    else:
        print('please enter a valid input')
