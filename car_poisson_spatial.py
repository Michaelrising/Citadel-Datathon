import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt
import geopandas as gpd
import pysal.lib as ps
from pymc3.distributions import continuous, distribution
from theano import scan, shared


class CAR(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    a : adjacency matrix
    w : weight matrix
    tau : precision at each location
    """

    def __init__(self, w, a, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau * tt.sum(w, axis=1)
        self.mode = 0.0

    def logp(self, x):
        tau = self.tau
        w = self.w
        a = self.a

        mu_w = tt.sum(x * a, axis=1) / tt.sum(w, axis=1)
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=tau).logp(x))

def car_likelihood(W, tau, alpha):
    def logp(value):
        # CAR prior likelihood
        deviance = value - alpha * np.matmul(W, value)
        norm_sq = np.matmul(deviance, deviance)
        logp = -0.5 * tau * norm_sq
        return logp

    return logp


grid_data = gpd.read_file('grid_data/grid_data.shp').drop('grid_index', axis=1)
data = grid_data[['crash_coun', 'black_perc', 'men_per_ar',
       'crime', 'nearest_po', 'investigat', 'traffic_vo', 'geometry']].copy()


# Compute the adjacency matrix (W) using queen contiguity

W = ps.weights.Queen.from_dataframe(data, geom_col='geometry')
W.transform = 'r'

# Initialize an n x n adjacency matrix filled with zeros
n = len(data)
adjacency_matrix = np.zeros((n, n))

# Fill the adjacency matrix with 1s for neighboring observations
for i, neighbors in W.neighbors.items():
    for j in neighbors:
        adjacency_matrix[i, j] = 1
W = W.full()[0]

with pm.Model() as car_spatial_poisson_model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sd=10)
    alpha_0 = pm.Normal("alpha_black", mu=0, sd=10)
    alpha_1 = pm.Normal("alpha_density", mu=0, sd=10)
    alpha_2 = pm.Normal("alpha_crime", mu=0, sd=10)
    alpha_3 = pm.Normal("alpha_police", mu=0, sd=10)
    alpha_4 = pm.Normal("alpha_investigation", mu=0, sd=10)
    alpha_5 = pm.Normal("alpha_traffic", mu=0, sd=10)

    # CAR prior
    tau = pm.Gamma("tau", alpha=1, beta=1)
    mu_phi = CAR('mu_phi', W, adjacency_matrix, tau)
    phi = pm.Deterministic("phi", mu_phi - tt.mean(mu_phi))

    # Linear regression
    mu = pm.math.exp(intercept +
                     alpha_0 * data['black_perc'] +
                    alpha_1 * data['men_per_ar'] +
                    alpha_2 * data['crime'] +
                    alpha_3 * data['nearest_po'] +
                    alpha_4 * data['investigat'] +
                    alpha_5 * data['traffic_vo'] +
                     phi)

    # Likelihood
    y_obs = pm.Poisson("crash_count", mu=mu, observed=data['crash_coun'])

with car_spatial_poisson_model:
    # Sample from the posterior
    trace = pm.sample(draws=2000, tune=1000, chains=2, target_accept=0.95, init='advi', cores=4, max_treedepth=15,
        return_inferencedata=True)

# Summarize the results
summary = pm.summary(trace).round(2)
print(summary)

# Plot the results
az.plot_trace(trace, var_names=['intercept', 'alpha_black', 'alpha_density', 'alpha_crime', 'alpha_police', 'alpha_investigation', 'alpha_traffic', 'tau', 'alpha'])
az.plot_forest(trace, var_names=['intercept', 'alpha_black', 'alpha_density', 'alpha_crime', 'alpha_police', 'alpha_investigation', 'alpha_traffic', 'tau', 'alpha'], combined=True, figsize=(10, 10))

