import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('ggplot')

##########################
#### Regression Plots ####
##########################

def plot_prior_and_posterior(prior, posterior, resolution=100):
    ws = np.linspace(-2, 2, num=resolution)
    bs = np.linspace(-2, 2, num=resolution)
    w, b = np.meshgrid(ws, bs)
    pos = np.empty(w.shape + (2,))
    pos[:, :, 0] = w; pos[:, :, 1] = b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title('Prior')
    ax1.contourf(w, b, prior.pdf(pos))
    ax1.set_xlabel("w"); ax1.set_ylabel("b")
    ax2.set_title('Posterior')
    ax2.contourf(w, b, posterior.pdf(pos))
    ax2.set_xlabel("w"); ax2.set_ylabel("b")
    return

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

plot_prior_and_posterior = plot_prior_and_posterior
plot_prior_and_posterior_predictions = plot_prior_and_posterior_predictions


#####################################
#### Binary Classification Plots ####
#####################################

def plot_logreg_data(X, y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Synthetic logreg Data')

    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.legend()
    ax.set_ylim([-3, 10])

    return fig, ax


def plot_logreg_prior_posterior(X, y, lam, plot_prior=True):
    '''
        Plot contours of the posterior for a two dimensional logistic regression problem.
    '''
    # Define a mesh for contour plotting the posterior
    w1, w2 = np.mgrid[-30:30:.5, -30:30:.5]
    pos = np.dstack((w1, w2))

    mu_prior = np.zeros([2])
    Sigma_prior = np.eye(2) / lam
    prior =  stats.multivariate_normal(mean = mu_prior,
                                 cov = Sigma_prior)

    W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))
    X_np = X.numpy()
    y_np = y.numpy()

    f = W @ X_np.T
    Log_Prior = np.log(stats.multivariate_normal.pdf(W, mean=np.zeros(2), cov=np.eye(2)/lam))
    Log_Like = np.sum(f * np.squeeze(y_np) - np.log(1+np.exp(f)), 1)
    Log_Joint = Log_Like + Log_Prior
    post = np.exp(Log_Joint - np.log(np.sum(np.exp(Log_Joint))))

    if plot_prior:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title('Prior')
        ax1.contourf(w1, w2, prior.pdf(pos))
        ax1.set_xlabel("theta_1"); ax1.set_ylabel("theta_2")


        ax2.set_title('Posterior')
        ax2.contourf(w1, w2, post.reshape((120, 120)))
        ax2.set_xlabel("theta_1"); ax2.set_ylabel("theta_2")
        ax2.set_xlim(-2, 20)
        ax2.set_ylim(-2, 20)
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
        ax1.set_title('Posterior')
        ax1.contourf(w1, w2, post.reshape((120, 120)))
        ax1.set_xlabel("theta_1"); ax1.set_ylabel("theta_2")
        ax1.set_xlim(-2, 20)
        ax1.set_ylim(-2, 20)
        ax2 = None

    return fig, ax1, ax2


def plot_cov_ellipse(subplot, mu, Sigma, nstd=1.5):

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs =  vals[order], vecs[:,order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, color='k', lw='4')
    subplot.plot(mu[0], mu[1], 'k*', label='mean')
    subplot.add_artist(ellip)
    return ellip

def plot_map_boundary(ax, theta):
    if theta[1] == 0:
        boundary = 0
    else:
        boundary = - theta[0] / theta[1]

    x1 = np.arange(-7,4)

    ax.plot(x1, boundary * x1, 'r', lw=2, alpha=0.5, label='Decision Boundary')
    ax.legend()

    return ax

def plot_la_vi_decision_boundaries(sample_model, num_samples, X, y, mu_la, L_la, mu_vi, L_vi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    X = X.numpy()
    y = y.numpy()
    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax1.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax1.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax2.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax2.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax1.set_xlabel("x_1"); ax1.set_ylabel("x_2")
    ax2.set_xlabel("x_1"); ax2.set_ylabel("x_2")

    for i in range(num_samples):
        theta_la, _ = sample_model(mu_la, L_la)
        plot_map_boundary(ax1, theta_la.detach().numpy())

    ax1.legend_.remove()
    ax1.set_title('Laplace Approx. Decision Boundaries')
    ax1.set_ylim([-3, 10])

    for i in range(num_samples):
        theta_vi, _ = sample_model(mu_vi, L_vi)
        plot_map_boundary(ax2, theta_vi.detach().numpy())

    ax2.legend_.remove()
    ax2.set_title('VI Decision Boundaries')
    ax2.set_ylim([-3, 10])
