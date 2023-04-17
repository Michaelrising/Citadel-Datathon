import arviz as az
import matplotlib.pyplot as plt
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


grid_data = gpd.read_file('grid_data/grid_data.shp')
# rename grid_data columns to crash_count, black_per, density, crime, police, investigation, traffic_volume, geometry
grid_data['crime'] = grid_data['crime'] / grid_data['crime'].max()
grid_data['investigat'] = grid_data['investigat'] / grid_data['investigat'].max()
grid_data['traffic_vo'] = grid_data['traffic_vo'] / grid_data['traffic_vo'].max()
grid_data['street_len'] = grid_data['street_len'] / grid_data['street_len'].max()
data = grid_data[['crash_coun', 'men_per_ar',
       'crime', 'nearest_po', 'investigat', 'traffic_vo', 'street_len', 'geometry']].copy()


# Compute the adjacency matrix (W) using queen contiguity

W = ps.weights.Queen.from_dataframe(data, geom_col='geometry')
# W.transform = 'r'
# W = ps.weights.Rook.from_dataframe(data, geom_col='geometry')
# W.transform = 'r'
# W = ps.weights.KNN.from_dataframe(data, geom_col='geometry', k=5)
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
    # alpha_0 = pm.Normal("alpha_black", mu=0, sd=10)
    alpha_1 = pm.Normal("alpha_density", mu=0, sd=10)
    alpha_2 = pm.Normal("alpha_crime", mu=0, sd=10)
    alpha_3 = pm.Normal("alpha_police", mu=0, sd=10)
    alpha_4 = pm.Normal("alpha_investigation", mu=0, sd=10)
    alpha_5 = pm.Normal("alpha_traffic", mu=0, sd=10)
    alpha_6 = pm.Normal("alpha_street", mu=0, sd=10)

    # CAR prior
    # Random effects (hierarchial) prior
    tau_h = pm.Gamma("tau_h", alpha=3.2761, beta=1.81)
    # Spatial clustering prior
    tau_c = pm.Gamma("tau_c", alpha=1.0, beta=1.0)

    # Regional random effects
    theta = pm.Normal("theta", mu=0.0, tau=tau_h, shape=len(data))
    # mu_phi = CAR("mu_phi", w=wmat, a=amat, tau=tau_c, shape=N)
    # tau = pm.Gamma("tau", alpha=1, beta=1)
    mu_phi = CAR('mu_phi', W, adjacency_matrix, tau_c, shape=len(data))
    phi = pm.Deterministic("phi", mu_phi) # - tt.mean(mu_phi)

    # Linear regression
    mu = pm.math.exp(intercept +
                     # alpha_0 * data['black_perc'] +
                    alpha_1 * data['men_per_ar'] +
                    alpha_2 * data['crime'] +
                    alpha_3 * data['nearest_po'] +
                    alpha_4 * data['investigat'] +
                    alpha_5 * data['traffic_vo'] +
                    alpha_6 * data['street_len'] +
                    theta + phi)

    # Likelihood using negative binomial
    alpha_ng = pm.Gamma("alpha", alpha=1, beta=1)
    y_obs = pm.NegativeBinomial("crash_count", mu=mu, alpha=alpha_ng,  observed=data['crash_coun'])

with car_spatial_poisson_model:
    # Sample from the posterior
    trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.95, cores=8, max_treedepth=15,
        return_inferencedata=True) # init='advi', random_seed=123

# Summarize the results
summary = pm.summary(trace).round(2)
print(summary)

# Plot the results
az.plot_trace(trace, var_names=['intercept',  'alpha_density', 'alpha_crime', 'alpha_police', 'alpha_investigation', 'alpha_traffic',  'alpha']) #'phi'
plt.show()
az.plot_forest(trace, var_names=['intercept', 'alpha_density', 'alpha_crime', 'alpha_police', 'alpha_investigation', 'alpha_traffic', 'tau', 'alpha'], combined=True, figsize=(10, 10))
plt.show()
# Plot the spatial effects
az.plot_forest(trace, var_names=['phi'], combined=True, figsize=(10, 10))
plt.show()

# # predict the crash count
# with car_spatial_poisson_model:


