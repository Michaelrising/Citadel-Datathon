import pandas as pd
import geopandas as gpd

import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from esda.moran import Moran
import pymc3 as pm
import contextily as ctx
from utils import convert_to_lat, convert_to_long

census = gpd.read_file('Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp')

crash_general = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')
crash_general = crash_general[crash_general['LATITUDE'].notna() & crash_general['LONGITUDE'].notna()]
crash_general['LATITUDE'] = crash_general['LATITUDE'].apply(convert_to_lat)
crash_general['LONGITUDE'] = crash_general['LONGITUDE'].apply(convert_to_long)
crash_points = gpd.points_from_xy(crash_general.LONGITUDE, crash_general.LATITUDE)
crash_gdf = gpd.GeoDataFrame(crash_general[['CRN', 'CRASH_YEAR']], geometry=crash_points)

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
crash_counts = gpd.sjoin(crash_gdf, grid, how='left', predicate='within').groupby(['index_right', 'CRASH_YEAR']).size().reset_index(name='count')
# if grid cell does not contain any crash, set the count to 0

crash_counts = crash_counts.set_index('index_right').reindex(range(len(grid))).fillna(0).reset_index()
ax = grid.plot(column=crash_counts['count'], cmap='Reds', figsize=(12, 12), edgecolor='gray', linewidth=0.2)
ax.set_title('Car Crash Count in Philadelphia by Grid Cells')
plt.show()

grid['crash_count'] = crash_counts['count']

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
        grid.loc[i, 'black_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_BLACK_NH'].values[0] / 100
        grid.loc[i, 'asian_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_ASIAN_NH'].values[0] / 100
        grid.loc[i, 'hispanic_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_HISPANIC'].values[0] /100
        grid.loc[i, 'men_per_area'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['men_per_area'].values[0]
    else:
        grid.loc[i, 'black_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_BLACK_NH'].mean()/100
        grid.loc[i, 'asian_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_ASIAN_NH'].mean()/100
        grid.loc[i, 'hispanic_percentage'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['PERCENT_HISPANIC'].mean()/100
        grid.loc[i, 'men_per_area'] = populations[populations.intersects(grid.loc[i, 'geometry'])]['men_per_area'].mean()

grid.fillna(0, inplace=True)
crime = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/crimes.csv')
crime = gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime.lng, crime.lat)).set_crs(epsg=4326)
crime = crime.to_crs(grid.crs)

crime_counts = gpd.sjoin(crime, grid, how='left', predicate='within').groupby('index_right').size().reset_index(name='count')
crime_counts = crime_counts.set_index('index_right').reindex(range(len(grid))).fillna(0).reset_index()
grid['crime'] = crime_counts['count']
# grid['crime'] = grid['crime'] / grid['crime'].sum()

police = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/police_stations.csv')
police = gpd.GeoDataFrame(police, geometry=gpd.points_from_xy(police.lng, police.lat)).set_crs(epsg=4326)
police = police.to_crs(grid.crs)

# calculate the distance to the nearest police station
nearest_police = []
for i in range(len(grid)):
    nearest_police.append(grid.loc[i, 'geometry'].distance(police.geometry).min())
grid['nearest_police'] = nearest_police


# traffic_stop = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/traffic_stops_philadelphia.csv')
# traffic_stop = traffic_stop[traffic_stop['lng'].notna() & traffic_stop['lat'].notna()]
# traffic_stop = gpd.GeoDataFrame(traffic_stop, geometry=gpd.points_from_xy(traffic_stop.lng, traffic_stop.lat)).set_crs(epsg=4326)
investigation = pd.read_csv('APAC_2023_Datasets/Traffic, Investigations _ Other/investigations.csv')
investigation = investigation[investigation['lng'].notna() & investigation['lat'].notna()]
investigation = gpd.GeoDataFrame(investigation, geometry=gpd.points_from_xy(investigation.lng, investigation.lat)).set_crs(epsg=4326)
investigation = investigation.to_crs(grid.crs)
investigation_count = gpd.sjoin(investigation, grid, how='left', predicate='within').groupby('index_right').size().reset_index(name='count')
investigation_count = investigation_count.set_index('index_right').reindex(range(len(grid))).fillna(0).reset_index()
grid['investigation'] = investigation_count['count']
# grid['investigation'] = grid['investigation'] / grid['investigation'].sum()

traffic_volume = pd.read_csv('Data/Traffic_Count_Locations.csv')
traffic_volume = traffic_volume[traffic_volume['X'].notna() & traffic_volume['Y'].notna()]
traffic_volume = traffic_volume[traffic_volume['setyear'] >= 2013]
traffic_volume = traffic_volume[traffic_volume['type'] == '15 min Volume']
traffic_volume = gpd.GeoDataFrame(traffic_volume, geometry=gpd.points_from_xy(traffic_volume.X, traffic_volume.Y)).set_crs(epsg=4326)
traffic_volume = traffic_volume.to_crs(grid.crs)
# filter each entry of traffic volume belongs to grid or not
traffic_volume = gpd.sjoin(traffic_volume, grid, how='left', predicate='within')
# group by grid index and calculate the average of traffic volume
traffic_volume = traffic_volume.groupby('index_right').mean().reset_index()
traffic_volume = traffic_volume[['index_right', 'recordnum']].set_index('index_right').reindex(range(len(grid))).fillna(traffic_volume['recordnum'].mean()).reset_index()

grid['traffic_volume'] = traffic_volume['recordnum'] #/traffic_volume['recordnum'].sum()

streets = gpd.read_file('Data/Street_Centerline/Street_Centerline.shp')#.set_crs(epsg=2272)
streets = streets.to_crs(grid.crs)
streets = streets[['LENGTH', 'geometry']]
# calculate the length of streets in each grid
grid["street_length"] = 0.0

# Loop through each grid cell
for i, grid_cell in grid.iterrows():
    # Clip streets by the current grid cell
    clipped_streets = gpd.clip(streets, grid_cell.geometry)

    # Calculate the total length of clipped streets within the grid cell
    street_length_in_cell = clipped_streets["LENGTH"].sum()

    # Update the grid cell's "street_length" value
    grid.loc[i, "street_length"] = street_length_in_cell

intersections = gpd.read_file('Data/Intersection_Controls-shp/a0bf09fb-47f9-4db3-935a-90fbb5956d0d2020330-1-zyti0a.69ghk.shp')

streets.plot(legend=True)
intersections.plot(ax=plt.gca(), color='red', markersize=1)
plt.show()

# grid['street_length'] = grid['street_length'] / streets['LENGTH'].sum()
grid.to_file('grid_data/grid_data.shp')

# plot all the features in one go (3 by 3 subplots) for visualization
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
grid.plot(column='traffic_volume', cmap='Blues', legend=True, ax=axs[0, 0])
axs[0, 0].set_title('Traffic Volume in Philadelphia by Grid Cells')
grid.plot(column='crime', cmap='Blues', legend=True, ax=axs[0, 1])
axs[0, 1].set_title('Crime Count in Philadelphia by Grid Cells')
grid.plot(column='investigation', cmap='Blues', legend=True, ax=axs[0, 2])
axs[0, 2].set_title('Investigation Count in Philadelphia by Grid Cells')
grid.plot(column='nearest_police', cmap='Blues', legend=True, ax=axs[1, 0])
axs[1, 0].set_title('Distance to Nearest Police Station in Philadelphia by Grid Cells')
grid.plot(column='black_percentage', cmap='Blues', legend=True, ax=axs[1, 1])
axs[1, 1].set_title('Percentage of Black Population in Philadelphia by Grid Cells')
grid.plot(column='asian_percentage', cmap='Blues', legend=True, ax=axs[1, 2])
axs[1, 2].set_title('Percentage of Asian Population in Philadelphia by Grid Cells')
grid.plot(column='hispanic_percentage', cmap='Blues', legend=True, ax=axs[2, 0])
axs[2, 0].set_title('Percentage of Hispanic Population in Philadelphia by Grid Cells')
grid.plot(column='men_per_area', cmap='Blues', legend=True, ax=axs[2, 1])
axs[2, 1].set_title('Men per area in Philadelphia by Grid Cells')
grid.plot(column='crash_count', cmap='Blues', legend=True, ax=axs[2, 2])
axs[2, 2].set_title('Crash count in Philadelphia by Grid Cells')
plt.show()


from scipy import  stats




