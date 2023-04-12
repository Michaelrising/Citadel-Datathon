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

census = gpd.read_file('Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp')

crash_general = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')
crash_general = crash_general[crash_general['LATITUDE'].notna() & crash_general['LONGITUDE'].notna()]
crash_general['LATITUDE'] = crash_general['LATITUDE'].apply(convert_to_lat)
crash_general['LONGITUDE'] = crash_general['LONGITUDE'].apply(convert_to_long)
crash_points = gpd.points_from_xy(crash_general.LONGITUDE, crash_general.LATITUDE)
crash_gdf = gpd.GeoDataFrame(geometry=crash_points)

populations = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/philadelphia_population_metrics.csv')
census['GEOID10'] = census['GEOID10'].astype(int)
populations = populations.merge(census[['GEOID10', 'geometry']], how='left', left_on='GEOGRAPHY_NAME', right_on='GEOID10')
populations = gpd.GeoDataFrame(populations)


philly_bbox = (-75.2803, 39.8718, -74.9558, 40.1376)

cell_size = 0.01  # in
x_min, y_min, x_max, y_max = philly_bbox
n_cells_x = int((x_max - x_min) / cell_size)
n_cells_y = int((y_max - y_min) / cell_size)
grid_cells = []
for x in range(n_cells_x):
    for y in range(n_cells_y):
        grid_cell = Polygon([(x_min + x * cell_size, y_min + y * cell_size),
                             (x_min + (x+1) * cell_size, y_min + y * cell_size),
                             (x_min + (x+1) * cell_size, y_min + (y+1) * cell_size),
                             (x_min + x * cell_size, y_min + (y+1) * cell_size)])
        grid_cells.append(grid_cell)

grid = gpd.GeoDataFrame({'geometry': grid_cells, 'index': range(len(grid_cells))})
grid.set_crs(epsg=4326, inplace=True)
grid = grid.to_crs(census.crs)
# first we filter out the grids that intersect with census
# 1
census_sindex = census.sindex
intersecting_idx = []
for geometry in grid.geometry:
    possible_matches_index = list(census_sindex.intersection(geometry.bounds))
    possible_matches = census.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(geometry)]
    if len(precise_matches) > 0:
        intersecting_idx.append(True)
    else:
        intersecting_idx.append(False)
gird_filtered = grid[intersecting_idx]
# 2
census_union = census.geometry.unary_union
intersects_census = grid.geometry.intersects(census_union)
gird_filtered = grid[intersects_census]

grid = gird_filtered.reset_index(drop=True).drop('index', axis=1)
# then we get the grids that contain crash
crash_gdf.set_crs(grid.crs, inplace=True)
# count the number of crash in each grid cell
crash_counts = gpd.sjoin(crash_gdf, grid, how='left', predicate='within').groupby('index_right').size().reset_index(name='count')
# if grid cell does not contain any crash, set the count to 0

crash_counts = crash_counts.set_index('index_right').reindex(range(len(grid))).fillna(0).reset_index()
ax = grid.plot(column=crash_counts['count'], cmap='Reds', figsize=(12, 12), edgecolor='gray', linewidth=0.2)
ax.set_title('Car Crash Count in Philadelphia by Grid Cells')
plt.show()

# we have the percentage of black in each census tract, we can calculate the percentage of black in each grid cell
# if the grid falls in one census tract, we can just use the percentage of black in that census tract;
# if the grid falls in two census tracts, we can use the percentage of black in both census tracts and average them

grid['black_percentage'] = 0
grid['asian_percentage'] = 0
grid['hispanic_percentage'] = 0
grid['men_per_area'] = 0
populations['men_per_area'] = populations['COUNT_ALL_RACES_ETHNICITIES'] / populations['Shape__Area'] * 10**3
for i in range(len(grid)):
    if populations.intersects(grid.loc[i, 'geometry']).sum() == 1:
        grid.loc[i, 'black_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_BLACK_NH'].values[0]
        grid.loc[i, 'asian_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_ASIAN_NH'].values[0]
        grid.loc[i, 'hispanic_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_HISPANIC'].values[0]
        grid['men_per_area'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['men_per_area'].values[0]
    else:
        grid.loc[i, 'black_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_BLACK_NH'].mean()
        grid.loc[i, 'asian_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_ASIAN_NH'].mean()
        grid.loc[i, 'hispanic_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_HISPANIC'].mean()
        grid.loc[i, 'men_per_area'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['men_per_area'].mean()



crime = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/crimes.csv')
crime = gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime.lng, crime.lat)).set_crs(epsg=4326)
crime = crime.to_crs(grid.crs)

crime_counts = gpd.sjoin(crime, grid, how='left', predicate='within').groupby('index_right').size().reset_index(name='count')
crime_counts = crime_counts.set_index('index_right').reindex(range(len(grid))).fillna(0).reset_index()
grid['crime'] = crime_counts['count']

police = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/police_stations.csv')
police = gpd.GeoDataFrame(police, geometry=gpd.points_from_xy(police.lng, police.lat)).set_crs(epsg=4326)
police = police.to_crs(grid.crs)

# calculate the distance to the nearest police station
nearest_police = []
for i in range(len(grid)):
    nearest_police.append(grid.loc[i, 'geometry'].distance(police.geometry).min())
grid['nearest_police'] = nearest_police
