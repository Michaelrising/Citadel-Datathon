import pandas as pd
import numpy as np
import seaborn as sns
import libpysal as ps
import spreg
from spreg.skater_reg import Skater_reg
import geopandas as gpd
import numpy as np
import seaborn as sns
from pysal.model import spreg
from pysal.lib import weights
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from esda.moran import Moran
from scipy import stats


grid = gpd.read_file('../grid_data/grid_data.shp').drop('grid_index', axis=1)
y = grid['crash_count'].values.reshape(-1) # log
# Set a small constant to handle zero values in the dependent variable, if necessary
constant = 1e-6
# Perform the Box-Cox transformation on the dependent variable (crash_count)
y_boxcox, lambda_boxcox = stats.boxcox(y + constant)

y = np.log(y)
y[y == -np.inf] = 0
factors = ['black_percentage','men_per_area', 'crime', 'nearest_police',
       'investigation']
# X = grid.drop(['crash_count', 'geometry', 'hispanic_percentage', 'asian_percentage', 'grid_index', 'traffic_volume'], axis=1).values
X = np.array([grid[var] for var in factors]).T
# X = (X - X.mean(axis=0)) / X.std(axis=0)

# Queen contiguity
w_queen = weights.contiguity.Queen.from_dataframe(grid.drop([ 'hispanic_percentage', 'asian_percentage', 'grid_index', 'traffic_volume'], axis=1))
w_rook = weights.contiguity.Rook.from_dataframe(grid.drop([ 'hispanic_percentage', 'asian_percentage', 'grid_index', 'traffic_volume'], axis=1))
# # Moran's I analysis for each variable
np.random.seed(123456)
# Calculate Moran's I for each variable
mi_results = [
    Moran(grid[variable], w_queen) for variable in factors
]
# Structure results as a list of tuples
mi_results = [
    (variable, res.I, res.p_sim)
    for variable, res in zip(factors, mi_results)
]
# Display on table
table = pd.DataFrame(
    mi_results, columns=["Variable", "Moran's I", "P-value"]
).set_index("Variable")

table

# K-nearest neighbors
k = 8  # Choose the number of nearest neighbors
w_knn = weights.distance.KNN.from_dataframe(grid.drop(['hispanic_percentage', 'asian_percentage', 'grid_index', 'traffic_volume'], axis=1), k=k)
w_kernel = weights.distance.Kernel.from_dataframe(grid.drop(['hispanic_percentage', 'asian_percentage', 'grid_index', 'traffic_volume'], axis=1), fixed=True, function = 'gaussian')

# Spatial Poisson model using the queen contiguity weights matrix
spatial_poisson_queen = spreg.GM_Lag(y, X, w=w_queen,  w_lags=1, spat_diag=True,name_y='crash_count', name_x=factors)
# Print the results for the queen contiguity weights matrix
print(spatial_poisson_queen.summary)


# Spatial Poisson model using the k-nearest neighbors weights matrix
spatial_poisson_knn = spreg.GM_Error(y, X, w=w_knn, name_y='crash_count', name_x=factors)

# Print the results for the k-nearest neighbors weights matrix
print(spatial_poisson_knn.summary)