
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'

# In[12]:


grid_data = gpd.read_file('grid_data/grid_data.shp')
grid_data = grid_data.rename(columns={'crash_coun': 'crash_count', 'black_perc': 'black_per', 'asian_perc': 'asian_per', 'hispanic_p':'hispanic_per',
                  'men_per_ar':'pop_density', 'crime':'crime_level', 'nearest_po':'police_dis', 'investigat':'invest_level', 'traffic_vo':"traffic_volume", 
                  'street_len':'street_len','geometry':'geometry'})
grid_data['crime_level'] = grid_data['crime_level'] / grid_data['crime_level'].max()
grid_data['invest_level'] = grid_data['invest_level'] / grid_data['invest_level'].max()
grid_data['traffic_volume'] = grid_data['traffic_volume'] / grid_data['traffic_volume'].max()
grid_data['street_len'] = grid_data['street_len'] / grid_data['street_len'].max()
grid_data['police_dis'] = grid_data['police_dis'] / grid_data['police_dis'].max()

# In[13]:


grid_data.columns


# In[18]:


# plot all the features in one go (3 by 3 subplots) for visualization
factor_list = [['black_per', 'asian_per', 'hispanic_per'], ['pop_density', 'street_len','traffic_volume'
       ], ['crime_level', 'police_dis', 'invest_level']]
fig, axs = plt.subplots(3, 3, figsize=(20, 20))
for i in range(3):
    for j in range(3):
        factor = factor_list[i][j]
        grid_data.plot(column=factor, cmap='Blues', legend=True, ax=axs[i, j])
        axs[i, j].set_title(factor + ' in Philadelphia by Grid Cells')
plt.savefig('figures/all_spatial_features.png', dpi=300)
plt.show()

grid_data.plot(column='crash_count', cmap='Blues', legend=True)
plt.title('Crash Count in Philadelphia by Grid Cells')
plt.savefig('figures/crash_count.png', dpi=300)
plt.show()


# In[19]:


import numpy as np
import seaborn as sns
# Compute the correlation matrix
corr_matrix = grid_data.drop(['geometry'], axis=1).corr()

# Create a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(np.round(corr_matrix, 2), annot=True, cmap="Blues")
# rotate the x-axis labels
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.savefig('figures/correlation_matrix_spatial_features.png', dpi=300)
plt.show()


# In[21]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[34]:


# For each X, calculate VIF and save in dataframe
X = grid_data.drop(['crash_count', 'geometry', 'traffic_volume', 'crime_level', 'pop_density'], axis=1) # , 'traffic_volume', 'pop_density', 'crime_level'
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[ ]:


data = grid_data.drop(['traffic_volume', 'crime_level', 'pop_density'], axis=1)
data.columns

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
from esda.moran import Moran



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


# Compute the adjacency matrix (W) using queen contiguity

W = ps.weights.Queen.from_dataframe(data, geom_col='geometry')
W.transform = 'r'
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

with pm.Model() as car_spatial_negabi_model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sd=10)
    alpha_0 = pm.Normal("alpha_black", mu=0, sd=10)
    alpha_1 = pm.Normal("alpha_asian", mu=0, sd=10)
    alpha_2 = pm.Normal("alpha_hispanic", mu=0, sd=10)
    alpha_3 = pm.Normal("alpha_police", mu=0, sd=10)
    alpha_4 = pm.Normal("alpha_investigation", mu=0, sd=10)
    # alpha_5 = pm.Normal("alpha_crime", mu=0, sd=10)
    alpha_6 = pm.Normal("alpha_street", mu=0, sd=10)

    # CAR prior
    # Random effects (hierarchial) prior
    tau_h = pm.Gamma("tau_h", alpha=3.2761, beta=1.81)
    # Spatial clustering prior
    tau_c = pm.Gamma("tau_c", alpha=1.0, beta=1.0)

    # Regional random effects
    theta = pm.Normal("theta", mu=0.0, tau=tau_h, shape=len(data))
    mu_phi = CAR('mu_phi', W, adjacency_matrix, tau_c, shape=len(data))
    phi = pm.Deterministic("phi", mu_phi ) #- tt.mean(mu_phi))

    # Linear regression
    mu = pm.math.exp(intercept +
                     alpha_0 * data['black_per'] +
                    alpha_1 * data['asian_per'] +
                    alpha_2 * data['hispanic_per'] +
                    alpha_3 * data['police_dis'] +
                    alpha_4 * data['invest_level'] +
                    # alpha_5 * data['crime_level'] +
                    alpha_6 * data['street_len'] +
                    theta + phi)

    # Likelihood using negative binomial
    alpha_ng = pm.Gamma("alpha", alpha=0.5, beta=0.5)
    y_obs = pm.NegativeBinomial("crash_count", mu=mu, alpha=alpha_ng,  observed=data['crash_count'])

with car_spatial_negabi_model:
    # Sample from the posterior
    trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.95, cores=1, max_treedepth=15,
        return_inferencedata=True, random_seed=123456)


# Summarize the results
summary = pm.summary(trace).round(2)
print(summary)

with car_spatial_negabi_model:
    ppc = pm.sample_posterior_predictive(
        trace, var_names=["intercept", 'alpha_police', 'alpha_street', 'alpha_investigation', 'crash_count'], random_seed=123456
    )

az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=car_spatial_negabi_model))
plt.xlim(0, 10000)
plt.show()

# Plot the results
az.plot_trace(trace, var_names=['intercept','alpha_black', 'alpha_asian', 'alpha_hispanic', 'alpha_police', 'alpha_investigation', 'alpha_street'], figsize=(10, 20)) #'phi'
plt.savefig('./figures/trace_parameters.png', dpi=300)
plt.show()
az.plot_forest(trace, var_names=['intercept', 'alpha_black',  'alpha_asian', 'alpha_hispanic', 'alpha_police', 'alpha_investigation', 'alpha_street'], combined=True, figsize=(10, 10))
plt.savefig('./figures/forest_parameters.png', dpi=300)
plt.show()

# Extract the parameter means
params_mean = summary["mean"]

# Calculate the sensitivity for each predictor variable
sensitivity = np.exp(params_mean.loc[["alpha_black", "alpha_police", "alpha_investigation", "alpha_crime", "alpha_street"]])

# Print the sensitivity
print("Sensitivity Analysis:")
print(sensitivity)

# Extract phi and theta from the trace
phi_trace = trace.posterior["phi"].values
theta_trace = trace.posterior["theta"].values

# Compute the average phi and theta values for each cell
phi_mean = np.mean(phi_trace, axis=(0, 1))
theta_mean = np.mean(theta_trace, axis=(0, 1))

# Compute the partial derivative of crash_count with respect to invest_level for each cell
invest_level_sensitivity = np.exp(params_mean.loc["alpha_investigation"] + theta_mean + phi_mean)

# Find the cell with the highest negative sensitivity
cell_idx = np.argmax(invest_level_sensitivity)

print(f"Cell with highest negative sensitivity for invest_level: {cell_idx}")


# Extract relevant parameter means
intercept_mean = params_mean.loc["intercept"]
alpha_investigation_mean = params_mean.loc["alpha_investigation"]


######## Hotspot identification ########
mu_crash_count = np.exp(intercept_mean +
                   params_mean.loc["alpha_black"] * data['black_per'] +
                   params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                   params_mean.loc["alpha_investigation"] * data['invest_level'] +
                   params_mean.loc["alpha_asian"] * data[ 'asian_per'] +
                    params_mean.loc["alpha_hispanic"] * data[ 'hispanic_per'] +
                   params_mean.loc["alpha_street"] * data[ 'street_len'] +
                   theta_mean+ phi_mean)
grid_data['pred_crash_count'] = mu_crash_count
# print resulst in one plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
grid_data.plot(column='pred_crash_count', ax=axs[0], legend=True,  cmap='Blues')
grid_data.plot(column='crash_count', ax=axs[1], legend=True, cmap='Blues')
axs[0].set_title("Predicted crash count by grid")
axs[1].set_title("Observed crash count by grid")
plt.savefig("./figures/pred_observed_crash_count.png", dpi=300)
plt.show()

# calculate R^2
from sklearn.metrics import r2_score
r2 = r2_score(mu_crash_count, data['crash_count'])
print(f"R^2: {r2}")


diff = data['crash_count'] - mu_crash_count
diff = diff.sort_values(ascending=False)
print(diff.head(10))

# point out the top 10 cells with highest difference in the map with red star with symbol size = 10

census = gpd.read_file("Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp")
crash_general = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')
crash_general = crash_general[crash_general['DEC_LAT'].notna() & crash_general['DEC_LONG'].notna()]
# crash_general['LATITUDE'] = crash_general['LATITUDE'].apply(convert_to_lat)
# crash_general['LONGITUDE'] = crash_general['LONGITUDE'].apply(convert_to_long)
crash_points = gpd.points_from_xy(crash_general.DEC_LONG, crash_general.DEC_LAT)
crash_gdf = gpd.GeoDataFrame(geometry=crash_points)

# count the car crashes in each census tract
crash_counts = gpd.sjoin(census, crash_gdf, op='contains').groupby(level=0).size().reset_index(name='count')
# if census tract does not contain any crash, set the count to 0
crash_counts = crash_counts.set_index('index').reindex(range(len(census))).fillna(0).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
census.plot(column=crash_counts['count'], cmap='Blues', alpha=0.8, ax=ax)
# grid_data.plot(column='pred_crash_count', ax=ax, legend=True,  cmap='Blues')
ax.set_title("Hot spots for car crash in Philadelphia")
for i in range(10):
    cell_idx = diff.index[i]
    cell = grid_data.loc[cell_idx]
    ax.annotate("*", xy=(cell.geometry.centroid.x, cell.geometry.centroid.y), color='red', size=20, label='Hot spot')
    # ax.annotate(f"{cell_idx}", xy=(cell.geometry.centroid.x, cell.geometry.centroid.y), color='red')
l0 = [plt.Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=10, label='Hot spot')]
plt.legend(l0, ['Hot spot'], loc='lower right')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig("./figures/hotspot_identification.png", dpi=300)
plt.show()

# ########## analyze the marginal effect of all factors on crash_count for all cells ##########
sensitivity_analyse = pd.DataFrame({"alphas": ['alpha_black', 'alpha_police', 'alpha_investigation', 'alpha_crime', 'alpha_street']})
effects = []
elasticity = []
# Compute the mean of crash_count
mean_data = data.mean(axis=0)

for alpha in ['alpha_black', 'alpha_police', 'alpha_investigation', 'alpha_crime', 'alpha_street']:
    effects.append(np.exp(summary.loc[alpha, "mean"]))
for factor in mean_data.index:
    mean_factor = mean_data[factor]
    delta_mean_factor = mean_factor * 0.1
    delta_crash_count = np.exp(delta_mean_factor* summary.loc[alpha, "mean"])
    elasticity.append()

sensitivity_analyse["marginal_effects"] = effects





mu_crash_count = pd.DataFrame({"crash_count": mu_crash_count}, index=data.index)
mu_crash_count = mu_crash_count["crash_count"].sort_values(ascending=False)
true_crash_count = data['crash_count'].sort_values(ascending=False)

main_cells = diff.index[:5].to_list() + true_crash_count.index[:5].to_list() + mu_crash_count.index[:5].to_list()
main_cells = np.unique(main_cells)

invest_level = data["invest_level"].copy()
total_invest_level = invest_level.sum()
##### strategy 1: average the investigation level in the main cells and the rest are set to 0
invest_level.loc[main_cells] = 0.2 * total_invest_level / len(main_cells)
# invest_level.loc[~invest_level.index.isin(main_cells)] = 0
strategy1_mu = np.exp(intercept_mean +
                          params_mean.loc["alpha_black"] * data['black_per'] +
                            params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                            alpha_investigation_mean * invest_level +
                            params_mean.loc["alpha_crime"] * data['crime_level'] +
                            params_mean.loc["alpha_street"] * data[ 'street_len'] +
                            theta_mean + phi_mean)
strategy1_crash_count = strategy1_mu.sum()

##### strategy 2: average the  80% investigation level in the main cells and the rest are set to 50%
invest_level = data["invest_level"].copy()
invest_level.loc[main_cells] = 0.4 * total_invest_level / len(main_cells)
# invest_level.loc[~invest_level.index.isin(main_cells)] = 0.2 * total_invest_level / len(invest_level.loc[~invest_level.index.isin(main_cells)])
strategy2_mu = np.exp(intercept_mean +
                          params_mean.loc["alpha_black"] * data['black_per'] +
                            params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                            alpha_investigation_mean * invest_level +
                            params_mean.loc["alpha_crime"] * data['crime_level'] +
                            params_mean.loc["alpha_street"] * data[ 'street_len'] +
                            theta_mean + phi_mean)
strategy2_crash_count = strategy2_mu.sum()

##### strategy 3: average the  50% investigation level in the main cells and the rest are set to 50%
invest_level = data["invest_level"].copy()
invest_level.loc[main_cells] = 0.6 * total_invest_level / len(main_cells)
# invest_level.loc[~invest_level.index.isin(main_cells)] = 0.5 * total_invest_level / len(invest_level.loc[~invest_level.index.isin(main_cells)])
strategy3_mu = np.exp(intercept_mean +
                          params_mean.loc["alpha_black"] * data['black_per'] +
                            params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                            alpha_investigation_mean * invest_level +
                            params_mean.loc["alpha_crime"] * data['crime_level'] +
                            params_mean.loc["alpha_street"] * data[ 'street_len'] +
                            theta_mean + phi_mean)
strategy3_crash_count = strategy3_mu.sum()

##### strategy 4: average the  20% investigation level in the main cells and the rest are set to 80%
invest_level.loc[main_cells] = 0.8 * total_invest_level / len(main_cells)
# invest_level.loc[~invest_level.index.isin(main_cells)] = 0.8 * total_invest_level / len(invest_level.loc[~invest_level.index.isin(main_cells)])
strategy4_mu = np.exp(intercept_mean +
                          params_mean.loc["alpha_black"] * data['black_per'] +
                            params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                            alpha_investigation_mean * invest_level +
                            params_mean.loc["alpha_crime"] * data['crime_level'] +
                            params_mean.loc["alpha_street"] * data[ 'street_len'] +
                            theta_mean + phi_mean)
strategy4_crash_count = strategy4_mu.sum()

strategies = gpd.GeoDataFrame({"strategy1": strategy1_mu, "strategy2": strategy2_mu, "strategy3": strategy3_mu, "strategy4": strategy4_mu}, geometry=data.geometry)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

strategies.plot(column="strategy1", ax=axs[0], cmap='Blues')
axs[0].set_title("20%")
strategies.plot(column="strategy2",  ax=axs[1], cmap='Blues')
axs[1].set_title("40%")
strategies.plot(column="strategy3", ax=axs[2], cmap='Blues')
axs[2].set_title("60%")
# with customized legend settings: size and location
strategies.plot(column="strategy4",ax=axs[3], cmap='Blues')
axs[3].set_title("80%")
# add the circle to the plot
for i in range(4):
    circle = plt.Circle((0.4, 0.35), 0.2, color='red', fill=False, transform=axs[i].transAxes)
    axs[i].add_artist(circle)
plt.savefig("./figures/investigation_strategies.png", dpi=300)
plt.show()
# Calculate crash count changes for the all cells
crash_decrease_rate = []
for i in range(1, 5):
    # Access the variable using the eval() function
    strategy_crash_count = eval(f'strategy{i}_crash_count')
    rate = (mu_crash_count.sum() - strategy_crash_count ) / mu_crash_count.sum()
    crash_decrease_rate.append(rate)



#### Spatial analysis ####

# Assume you have a GeoDataFrame with the same index as your data, called `gdf`
grid_data["theta_mean"] = trace.posterior["theta"].mean(dim=("chain", "draw")).values
grid_data["phi_mean"] = trace.posterior["phi"].mean(dim=("chain", "draw")).values

# Plot the posterior means of theta and phi
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

grid_data.plot(column="theta_mean", cmap="Blues", legend=True, ax=ax1)
ax1.set_title("Posterior mean of theta (Regional random effects)")

grid_data.plot(column="phi_mean", cmap="Blues", legend=True, ax=ax2)
ax2.set_title("Posterior mean of phi (Spatial clustering effects)")
plt.savefig("./figures/spatial_random_effects.png", dpi=300)
plt.show()

# Compute Moran's I for the phi posterior mean
phi_mean = trace.posterior["phi"].mean(dim=("chain", "draw")).values
moran = Moran(phi_mean, W)

print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")

# Define the range of invest_level changes
invest_level_changes = np.linspace(data['invest_level'].min(), data['invest_level'].max(), 100)
# Calculate crash count changes for the selected cell
crash_count_changes = []

for change in invest_level_changes:
    new_invest_level = change #max(data.loc[cell_idx, "invest_level"], ) * (1 + change)
    invest_level.loc[diff.index[:10]] = new_invest_level
    mu_change = np.exp(intercept_mean +
                       params_mean.loc["alpha_black"] * data['black_per'] +
                       params_mean.loc["alpha_police"] * data[ 'police_dis'] +
                       alpha_investigation_mean * invest_level +
                       params_mean.loc["alpha_crime"] * data['crime_level'] +
                       params_mean.loc["alpha_street"] * data[ 'street_len'] +
                       theta_mean + phi_mean)
    crash_count_changes.append(mu_change.sum())

# Plot the changes in crash counts as a function of the change in invest_level
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(invest_level_changes * 100, (1-crash_count_changes / mu_crash_count.sum())*100)
plt.xlabel("Investigation Level in Hot Spots (%)", size=13)
plt.ylabel("Crash decline by percentage (%)", size=13)
plt.title("Crash Frequency Change vs Investigation Level", size=14)
plt.grid()
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig("./figures/count_frequency_investigation_level_change.png", dpi=300)
plt.show()
