import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
from pysal.model import spreg
from pysal.lib import weights
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from utils import convert_to_lat, convert_to_long

philly_bbox = (-75.2803, 39.8718, -74.9558, 40.1376)

crash_general = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')
crash_general = crash_general[crash_general['LATITUDE'].notna() & crash_general['LONGITUDE'].notna()]
crash_general['LATITUDE'] = crash_general['LATITUDE'].apply(convert_to_lat)
crash_general['LONGITUDE'] = crash_general['LONGITUDE'].apply(convert_to_long)
crash_points = gpd.points_from_xy(crash_general.LONGITUDE, crash_general.LATITUDE)
crash_gdf = gpd.GeoDataFrame(geometry=crash_points)

# count the car crashes in each census tract
census = gpd.read_file('Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp')
crash_counts = gpd.sjoin(census, crash_gdf, op='contains').groupby(level=0).size().reset_index(name='count')
# if census tract does not contain any crash, set the count to 0
crash_counts = crash_counts.set_index('index').reindex(range(len(census))).fillna(0).reset_index()
fig, ax = plt.subplots(figsize=(10,10))
census.plot(column=crash_counts['count'], cmap='Reds', legend=True, ax=ax)
ax.set_title('Car Crash Count in Philadelphia by Census Tracts')
plt.show()


# in each grid, extract the other features tha may be related to the crash count
# e.g. population, income, etc.
populations = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/philadelphia_population_metrics.csv')
census['GEOID10'] = census['GEOID10'].astype(int)
populations = populations.merge(census[['GEOID10', 'geometry']], how='left', left_on='GEOGRAPHY_NAME', right_on='GEOID10')
populations = gpd.GeoDataFrame(populations)
fig6, ax = plt.subplots(figsize=(10,10))
populations.plot(column='PERCENT_ASIAN_NH', cmap='OrRd', legend=True, ax=ax)
ax.set_title('Population of black percentage')
plt.show()


crime = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/crimes.csv')
crime = gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime.lng, crime.lat))
crime_count = gpd.sjoin(census, crime, predicate='contains').groupby(level=0).size().reset_index(name='count')
crime_count = crime_count.set_index('index').reindex(range(len(census))).fillna(0).reset_index()
fig7, ax = plt.subplots(figsize=(10,10))
census.plot(column=crime_count['count'], cmap='Reds', legend=True, ax=ax)
ax.set_title('Crime Count in Philadelphia by Census Tracts')
plt.show()

# now we have car crash count, population, crime count in each census tract
# we can use these featrues to train a model to predict the car crash count in each census tract
# merge the features into one geodataframe

data = gpd.GeoDataFrame(geometry=census['geometry'])
data['crash_count'] = crash_counts['count']
data['crime'] = crime_count['count'] # normalize count
data['crime'] = data['crime'] / data['crime'].max()
data['black_percentage'] = populations['PERCENT_BLACK_NH']
data['asian_percentage'] = populations['PERCENT_ASIAN_NH']
data['hispanic_percentage'] = populations['PERCENT_HISPANIC']

y = np.log(data['crash_count'].values)
X = data.drop(['crash_count', 'geometry'], axis=1).values

# Queen contiguity
w_queen = weights.contiguity.Queen.from_dataframe(data)

# # Moran's I analysis for each variable
# np.random.seed(123456)
# # Calculate Moran's I for each variable
# mi_results = [
#     Moran(data[variable], w_queen) for variable in data.columns
# ]
# # Structure results as a list of tuples
# mi_results = [
#     (variable, res.I, res.p_sim)
#     for variable, res in zip(data.columns, mi_results)
# ]
# # Display on table
# table = pd.DataFrame(
#     mi_results, columns=["Variable", "Moran's I", "P-value"]
# ).set_index("Variable")

table

# K-nearest neighbors
k = 4  # Choose the number of nearest neighbors
w_knn = weights.distance.KNN.from_dataframe(data, k=k)

# Spatial Poisson model using the queen contiguity weights matrix
spatial_poisson_queen = spreg.GM_Lag(y, X, w=w_queen, name_y='crash_count', name_x=['crime', 'black_percentage', 'asian_percentage', 'hispanic_percentage'])

# Spatial Poisson model using the k-nearest neighbors weights matrix
spatial_poisson_knn = spreg.GM_Lag(y, X, w=w_knn, name_y='crash_count', name_x=['crime', 'black_percentage', 'asian_percentage', 'hispanic_percentage'])

# Print the results for the queen contiguity weights matrix
print(spatial_poisson_queen.summary)

# Print the results for the k-nearest neighbors weights matrix
print(spatial_poisson_knn.summary)


