# @Author: aaronmishkin
# @Date:   2018-06-23T02:43:22+02:00
# @Email:  aaron.mishkin@riken.jp
# @Last modified by:   aaronmishkin
# @Last modified time: 2018-06-24T19:55:01+02:00


import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('ggplot')

##########################
#### Regression Plots ####
##########################

def lr_visualize_prior_posterior(w0,w1,prior_density,numerical_posterior_density):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ## IMPORTANT: The scale for the prior/joint countour plots is only 1/4 of the posterior countour's scale. ##
    posterior_levels = np.linspace(0, 1.6, 10)
    prior_levels = np.linspace(0, 0.4, 10)
    # Plot Prior and Posterior
    c1 = plot_density(ax1, w1, w0, prior_density, prior_levels, title="Prior", xlim=[-2,2], ylim=[-2,2])
    c2 = plot_density(ax2, w1, w0, numerical_posterior_density, posterior_levels, title="Posterior", xlim=[-2,2], ylim=[-2,2])

    cb = fig.colorbar(c1, ax=ax1, ticks=[0, 0.1, 0.2, 0.3, 0.4])
    cb = fig.colorbar(c2, ax=ax2, ticks=[0, 0.4, 0.8, 1.2, 1.6])
    return

def lr_visualize_numerical_vs_closed_form_posteriors(exact_posterior_density, numerical_posterior_density, w1,w0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    posterior_levels = np.linspace(0, 1.6, 10)

    c1 = plot_density(ax1, w1, w0, numerical_posterior_density, posterior_levels, title="Numerical Posterior", xlim=[-2,2], ylim=[-2,2])
    c2 = plot_density(ax2, w1, w0, exact_posterior_density, posterior_levels, title="Closed-Form Posterior", xlim=[-2,2], ylim=[-2,2])

    cb = fig.colorbar(c1, ax=ax1, ticks=[0, 0.4, 0.8, 1.2, 1.6])
    cb = fig.colorbar(c2, ax=ax2, ticks=[0, 0.4, 0.8, 1.2, 1.6])


def plot_prior_and_posterior_predictions(x, y, x_pred, prior_preds, posterior_preds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title('Prior Predictions')
    ax1.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
    ax1.plot(x_pred, prior_preds.T, 'r', lw=2, alpha=0.5, label='predictions')
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax2.set_title('Posterior Predictions')
    ax2.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
    ax2.plot(x_pred, posterior_preds.T, 'r', lw=2, alpha=0.5, label='predictions')
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-15, 15])
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-15, 15])

def plot_toy_data(x, y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Toy Data')
    ax.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()

def plot_training(losses):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Training Curve")
    ax.plot(losses)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Negative ELBO")

def plot_predictions(x, y, x_pred, mu_pred):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Predictions')
    ax.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
    ax.plot(x_pred, mu_pred[0].T, 'r-', alpha=0.5, label='Predictions')
    ax.plot(x_pred, mu_pred[1:].T, 'r-', alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()

##################################################
#### Laplace Approximation Plotting Functions ####
##################################################
def la_visualize_prior_joint_posterior(w1,w2,prior_density,joint_density,posterior_density):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    prior_joint_levels = np.linspace(0, 0.004, 20)
    posterior_levels = np.linspace(0, 0.016, 20)

    c1 = plot_density(ax1, w1, w2, prior_density, prior_joint_levels, title="Prior", xlim=[-20,20], ylim=[-20,20])
    c2 = plot_density(ax2, w1, w2, joint_density, prior_joint_levels, title="Joint", xlim=[-20,20], ylim=[-20,20])
    c3 = plot_density(ax3, w1, w2, posterior_density, posterior_levels, title="Posterior", xlim=[-20,20], ylim=[-20,20])


    cb = fig.colorbar(c2, ax=ax2, ticks=[0, 0.001, 0.002, 0.003, 0.004])
    cb = fig.colorbar(c3, ax=ax3, ticks=[0, 0.004, 0.008, 0.012, 0.016])

    return

def visualize_MAP_estimate(objective_history, w_history, max_iters, w1, w2, posterior_density, w_map, X, y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_objective_history(ax1, range(max_iters), objective_history, 'MAP Objective')
    ax1.set_ylim(0,10)

    plot_logreg_data(ax2, X.numpy(), y.numpy())
    plot_map_boundary(ax2, w_map.detach().numpy())
    ax2.set_title('MAP Decision Boundary')

    c1 = plot_density(ax3, w1, w2, posterior_density, None, title="Posterior")
    w_np = w_history.detach().numpy()
    ax3.plot(w_np[:,0],w_np[:,1], 'k.', ms=4, label='Training Path')
    ax3.plot(w_map[0].detach().numpy(), w_map[1].detach().numpy(), 'ro', ms=10, label='MAP Estimate')
    ax3.legend()
    fig.colorbar(c1)
    print('The MAP estimate is:', w_map.detach().numpy())

    return

def visualize_laplace_approximation(w1,w2,posterior_density,mu_la,Sigma_la):

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

    c1 = plot_density(ax1, w1, w2, posterior_density, None, title="Posterior")
    ellipse = plot_cov_ellipse(ax1, mu_la.detach().numpy(), Sigma_la.detach().numpy(), label='Laplace Approx.')
    ax1.legend()
    t = ax1.set_title('Laplace Approximation')
    cb = fig.colorbar(c1)

##################################################
#### Variational Inference Plotting Functions ####
##################################################
### Visualizing the Variational Approximation ###

def visualize_variational_approximation(objective_history,mu_history,max_iters,w1,w2,posterior_density,mu_vi,Sigma_vi,X,y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_objective_history(ax1, range(max_iters), objective_history, 'Negative ELBO')

    plot_logreg_data(ax2, X.numpy(), y.numpy())
    plot_map_boundary(ax2, mu_vi.detach().numpy())
    ax2.set_title('VI Decision Boundary (Mean of Distribution)')

    c1 = plot_density(ax3, w1, w2, posterior_density, None, title="Posterior")
    mu_np = mu_history.detach().numpy()
    ax3.plot(mu_np[:,0],mu_np[:,1], 'k.', ms=4, label='Training Path')
    plot_cov_ellipse(ax3, mu_vi.detach().numpy(), Sigma_vi.detach().numpy(), label='VI Approx.', color='k')

    ax3.legend()
    t = ax3.set_title('Variational Inference')
    cb = fig.colorbar(c1)

def visualize_VI_vs_LA(w1,w2,posterior_density,sample_model,mu_la,L_la,Sigma_la, mu_vi,L_vi,Sigma_vi,X,y):
    num_samples = 10

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_decision_boundaries(ax1, sample_model, num_samples, X, y, mu_la, L_la, 'Laplace Approx. Decision Boundaries')
    plot_decision_boundaries(ax2, sample_model, num_samples, X, y, mu_vi, L_vi, 'VI Decision Boundaries')

    c1 = plot_density(ax3, w1, w2, posterior_density, None, title="Posterior")
    plot_cov_ellipse(ax3, mu_la.detach().numpy(), Sigma_la.detach().numpy(), label='Laplace Approx.', color='r')
    plot_cov_ellipse(ax3, mu_vi.detach().numpy(), Sigma_vi.detach().numpy(), label='VI Approx.', color='k')
    cb = fig.colorbar(c1)
    ax3.legend()
    t = ax3.set_title('Comparison of Laplace and VI')

##########################
#### Helper Functions ####
##########################

def plot_objective_history(ax, iterations, objective_history, ylabel):
    ax.plot(list(iterations), objective_history, 'kx', alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Iterations')
    ax.set_title("Training Loss")


def plot_logreg_data(ax, X, y):
    ax.set_title('Synthetic logreg Data')

    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.legend()
    ax.set_ylim([-3, 10])

def plot_density(ax, w1, w2, density, levels, title="", xlim=[-2,20], ylim=[-2,20]):
    ax.set_title(title)
    contour = ax.contourf(w1, w2, density, levels=levels)
    ax.set_xlabel("w_1"); ax.set_ylabel("w_2")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return contour

def plot_cov_ellipse(ax, mu, Sigma, nstd=1.5, label='', color='r'):

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs =  vals[order], vecs[:,order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, color=color, lw='4')
    ax.plot(mu[0], mu[1], 'o', color=color, label=label)
    ax.add_artist(ellip)
    return ellip

def plot_map_boundary(ax, w):
    if w[1] == 0:
        boundary = 0
    else:
        boundary = - w[0] / w[1]

    x1 = np.arange(-7,4)

    ax.plot(x1, boundary * x1, 'r', lw=2, alpha=0.5, label='Decision Boundary')
    ax.legend()

    return ax

def plot_decision_boundaries(ax, sample_model, num_samples, X, y, mu, L, title):
    X = X.numpy()
    y = y.numpy()
    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")

    for i in range(num_samples):
        w, _ = sample_model(mu, L)
        plot_map_boundary(ax, w.detach().numpy())

    ax.legend_.remove()
    ax.set_title(title)
    ax.set_ylim([-3, 10])
