import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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