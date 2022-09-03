

# INSTRUCTIONS:
# Jump to section "if __name__ == "__main__":" at bottom of this file.


from ellie_plotting import *
from Uncertainty import *
import matplotlib.pyplot as plt
import pandas as pd

def grid_search():
	''' This function implements a grid search to compute the posterior over a and b.

		Returns:
		--------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
	'''

	# Define parameter ranges for the grid search
	a_best = 18.1
	b_best = 0.026

	# Number of values considered for each parameter within a given interval
	N = 20

	# Vectors of parameter values
	a = np.linspace(a_best/2,a_best*1.5, N)
	b = np.linspace(b_best/2,b_best*1.5, N)

	# Grid of parameter values: returns every possible combination of parameters in a and b
	A, B = np.meshgrid(a, b, indexing='ij')

	# Empty 2D matrix for objective function
	S = np.zeros(A.shape)

	# Data for calibration
	tp,po = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T

	# Error variance
	v = 0.01

	# Grid search algorithm
	for i in range(len(a)):
		for j in range(len(b)):
			# Compute the sum of squares objective function at each value
			t, pm = solve_lpm(lpm, 2009, 2019, 0.25, 25.16, [10.87, a[i], b[j]])
			S[i,j] = np.sum((po-pm)**2)/v

	# Compute the posterior
	P = np.exp(-S/2.)

	# Normalize to a probability density function
	Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])
	P = P/Pint

	# Plot posterior parameter distribution
	# plot_posterior(a, b, P=P)

	return a,b,P

def construct_samples(a,b,P,N_samples):
	''' This function constructs samples from a multivariate normal distribution
	    fitted to the data.

		Parameters:
		-----------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''

	# Compute properties (fitting) of multivariate normal distribution
	# Mean = a vector of parameter means
	# Covariance = a matrix of parameter variances and correlations
	A, B = np.meshgrid(a,b,indexing='ij')
	mean, covariance = fit_mvn([A,B], P)

	# Create samples using numpy function multivariate_normal (Google it)
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	# Plot samples and predictions
	# plot_samples(a, b, P=P, samples=samples)

	return samples


def model_ensemble(samples):
	''' Runs the model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# Run your parameter samples through the model and plot the predictions.

	# 1. choose a time vector to evaluate your model between 1953 and 2012
	# t =
	t = np.linspace(2009, 2019, 101)

	# 2. create a figure and axes (see TASK 1)
	# f,ax =
	f, ax = plt.subplots(1, 1)

	# 3. for each sample, solve and plot the model  (see TASK 1)
	for a, b in samples:
		# pm=
		# ax.plot(
		# *hint* use lw= and alpha= to set linewidth and transparency
		t, pm = solve_lpm(lpm, 2009, 2019, 0.1, 25.16, [10.87, a, b])
		ax.plot(t, pm, 'k-', lw=0.25, alpha=0.2)
	ax.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')

	# get the data
	tp, po = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
	ax.axhline(25.16, color='g', linestyle=':', label='overpressure')

	# 4. plot Wairakei data as error bars
	# *hint* see TASK 1 for appropriate plotting commands
	v = 0.05
	ax.errorbar(tp, po, yerr=v, fmt='ro', label='data')
	ax.set_xlabel('Time (years)')
	ax.set_ylabel('Pressure (MPa)')
	ax.legend(loc = 'upper left', prop={'size': 6})
	plt.show()

def model_ensemble_with_forecasts(samples):
	''' Runs the model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''

	fig1, ax1 = plt.subplots()
	tp,po = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T

	# For each sample, solve and plot the model for each scale
	for a,b in samples:
		t1, pm = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b])
		t1, pm3 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale = 2.)
		t1, pm4 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale = 1.2)
		t1, pm5 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale = 1.5)

		timelength = len(t1)
		middle_index = int(timelength/2)

		ax1.plot(t1, pm,'k-', lw=0.25,alpha=0.2)
		ax1.plot(t1[middle_index:], pm3[middle_index:], 'b-', lw=0.25, alpha=0.2)
		ax1.plot(t1[middle_index:], pm4[middle_index:], 'm-', lw=0.25, alpha=0.2)
		ax1.plot(t1[middle_index:], pm5[middle_index:], 'y-', lw=0.25, alpha=0.2)

	ax1.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')
	ax1.plot([],[],'b-', lw=0.5,alpha=0.4, label='scale: 2')
	ax1.plot([],[],'m-', lw=0.5,alpha=0.4, label='scale: 1.2')
	ax1.plot([],[],'y-', lw=0.5,alpha=0.4, label='scale: 1.5')
	ax1.axvline(2019, color='b', linestyle=':', label='calibration/forecast')
	ax1.axhline(25.16, color='g', linestyle=':', label='overpressure')

	v = 0.01
	ax1.errorbar(tp, po, yerr=v, fmt='ro', markersize=2, label='data')
	ax1.set_title('Pressure Variation in Ahuroa Resevoir')
	ax1.set_xlabel('Time (years)')
	ax1.set_ylabel('Pressure (MPa)')
	ax1.legend(prop={'size': 6})

	plt.savefig('Uncertainty plot 1')
	plt.show()

def leakage_forecasting(samples):
	''' Runs the model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''

	fig2, ax2 = plt.subplots()
	overpressure = 25.16

	# For each sample, solve and plot the model for each scale
	for a,b in samples:
		t1, pm = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b])
		t1, pm3 = solve_lpm(lpm, 2019, 2029, 0.1, 25.16, [10.87, a, b], scale = 2.) #blueplot
		t1, pm4 = solve_lpm(lpm, 2019, 2029, 0.1, 25.16, [10.87, a, b], scale = 1.2) #magentaplot
		t1, pm5 = solve_lpm(lpm, 2019, 2029, 0.1, 25.16, [10.87, a, b], scale = 1.5) #yellowplot

		timelength = len(t1)
		middle_index = int(timelength/2)

		dleakage1 = gasLeakage(t1, pm, overpressure, b)
		cumulLeak1 = integralFunc(t1, dleakage1)
		cumulLeak1 = np.array(cumulLeak1.tolist())

		dleakage3 = gasLeakage(t1, pm3, overpressure, b)
		cumulLeak3 = integralFunc(t1, dleakage3)
		cumulLeak3 = cumulLeak3.tolist()
		cumulLeak3 = [x + 0.4 for x in cumulLeak3]

		dleakage4 = gasLeakage(t1, pm4, overpressure, b)
		cumulLeak4 = integralFunc(t1, dleakage4)
		cumulLeak4 = cumulLeak4.tolist()
		cumulLeak4 = [x + 0.036 for x in cumulLeak4]

		dleakage5 = gasLeakage(t1, pm5, overpressure, b)
		cumulLeak5 = integralFunc(t1, dleakage5)
		cumulLeak5 = cumulLeak5.tolist()
		cumulLeak5 = [x + 0.12 for x in cumulLeak5]

		# ax2.plot(t1,cumulLeak1,'k-', lw=0.25,alpha=0.2)
		# ax2.plot(t1[middle_index:],cumulLeak3[middle_index:],'b-', lw=0.25,alpha=0.2)
		# ax2.plot(t1[middle_index:],cumulLeak4[middle_index:],'m-', lw=0.25,alpha=0.2)
		# ax2.plot(t1[middle_index:],cumulLeak5[middle_index:],'y-', lw=0.25,alpha=0.2)

		ax2.plot(t1, cumulLeak1, 'k-', lw=0.25, alpha=0.2)
		ax2.plot(t1, cumulLeak3, 'b-', lw=0.25, alpha=0.2)
		ax2.plot(t1, cumulLeak4, 'm-', lw=0.25, alpha=0.2)
		ax2.plot(t1, cumulLeak5, 'y-', lw=0.25, alpha=0.2)

	ax2.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')
	ax2.plot([],[],'b-', lw=0.5,alpha=0.4, label='scale: 2')
	ax2.plot([],[],'m-', lw=0.5,alpha=0.4, label='scale: 1.2')
	ax2.plot([],[],'y-', lw=0.5,alpha=0.4, label='scale: 1.5')
	ax2.axvline(2019, color='b', linestyle=':', label='calibration/forecast')

	ax2.set_title('Pressure change in resevoir due to Methane Leakage')
	ax2.set_xlabel('Time (years)')
	ax2.set_ylabel('Pressure (MPa)')
	ax2.legend(prop={'size': 6})

	plt.savefig('Uncertainty plot 2')
	plt.show()

def plot_histograms(samples):

	plt.rcParams["figure.figsize"] = [7.50, 3.50]
	plt.rcParams["figure.autolayout"] = True

	fig3, ax3 = plt.subplots()

	# Extracting parameter a samples from the multivariate normal
	x1 = samples[:, 0]
	x1 = x1.tolist()

	# Converting list of parameter values into dataframe
	df1 = pd.DataFrame(x1)

	# Creating empty histogram for plotting frequency density
	num_bins = 30
	n, bins, patches = plt.hist(x1, num_bins)

	# Finding and plotting 95% confidence interval marker for parameter a
	lower = df1.quantile(0.05)
	upper = df1.quantile(0.95)
	ax3.axvline(lower[0], color='r', linestyle=':')
	ax3.axvline(upper[0], color='r', linestyle=':')

	# PLotting axes title and legend
	ax3.set_title("Frequency Density plot for Parameter a")
	ax3.set_xlabel('Parameter a')
	ax3.set_ylabel('Frequency density')
	plt.savefig('Histogram a')
	plt.show()

	fig4, ax4 = plt.subplots()
	x2 = samples[:, 1]
	x2 = x2.tolist()
	df2 = pd.DataFrame(x2)

	# Creating empty histogram for plotting frequency density
	num_bins = 30
	n, bins, patches = plt.hist(x2, num_bins)

	# Finding and plotting 95% confidence interval marker for parameter d
	lower = df2.quantile(0.05)
	upper = df2.quantile(0.95)
	ax4.axvline(lower[0], color='r', linestyle=':')
	ax4.axvline(upper[0], color='r', linestyle=':')

	# PLotting axes title and legend
	ax4.set_title("Frequency Density plot for Parameter d")
	ax4.set_xlabel('Parameter d')
	ax4.set_ylabel('Frequency density')
	plt.savefig('Histogram b')
	plt.show()


if __name__=="__main__":

	a,b,posterior = grid_search()
	N = 50
	samples = construct_samples(a, b, posterior, N)

	plot_histograms(samples)
	#model_ensemble(samples)
	#model_ensemble_with_forecasts(samples)
	leakage_forecasting(samples)


