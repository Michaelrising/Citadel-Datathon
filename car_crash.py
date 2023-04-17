import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
from pysal.model import spreg
from pysal.lib import weights
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from esda.moran import Moran
import pymc3 as pm
import contextily as ctx
from pymining import itemmining, assocrules, perftesting

# car crash data sets
grid = gpd.read_file('grid_data/grid_data.shp')

crash_info_commercials = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_commericial_vehicles.csv')
crash_flag_info = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_flag_variables.csv')
crash_info_motor = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_motorcycle.csv')
crash_info_trialed = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_trailed_vehicles.csv')
crash_info_vehicle = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_vehicles.csv')
crash_info_roadway = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_roadway.csv')
crash_info_people = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_people.csv')
crash_general = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')

# merge all the crash related data sets by CRN
crash_info = crash_info_commercials.merge(crash_flag_info, on='CRN', how='left').\
                                    merge(crash_info_motor, on='CRN', how='left').\
                                    merge(crash_info_trialed, on='CRN', how='left').\
                                    merge(crash_info_vehicle, on='CRN', how='left').\
                                    merge(crash_info_roadway, on='CRN', how='left').\
                                    merge(crash_info_people, on='CRN', how='left').\
                                    merge(crash_general, on='CRN', how='left').reset_index(drop=True)




traffic_dist=gpd.read_file('Data/Highway_Districts-shp/Highway_Districts_arc-shp/f7463796-30e2-4d7f-a11a-e21475ddde6c202041-1-1hepqr9.ov2p.shp')
# plot the traffic distribution
traffic_dist.plot()
plt.show()


crash_severity = crash_general[['CRN', 'CRASH_YEAR', 'CRASH_MONTH', 'MAX_SEVERITY_LEVEL', 'INJURY_COUNT', 'ILLUMINATION', 'WEATHER1', 'ROAD_CONDITION', 'LATITUDE', 'LONGITUDE']].dropna()
crash_severity = crash_severity.merge(crash_info_vehicle[['CRN', 'TRAVEL_SPD', 'DVR_PRES_IND']], on='CRN', how='left')
# drop the speed == 0
crash_severity = crash_severity[crash_severity['TRAVEL_SPD'] != 0]
crash_severity.loc[crash_severity['TRAVEL_SPD'] > 100, 'TRAVEL_SPD'] = 100
crash_severity['TRAVEL_SPD'].fillna(crash_severity['TRAVEL_SPD'].mean(), inplace=True)
crash_severity['TRAVEL_SPD'] = crash_severity['TRAVEL_SPD'] / crash_severity['TRAVEL_SPD'].max()
crash_severity.dropna(inplace=True)

# map the car crash to the grid, with grid information for each crash
crash_severity = gpd.GeoDataFrame(crash_severity, geometry=gpd.points_from_xy(crash_severity.LONGITUDE, crash_severity.LATITUDE)).set_crs(epsg=4326)
crash_severity = crash_severity.to_crs(grid.crs)
merged_crash_severity = gpd.sjoin(crash_severity, grid, how='left', predicate='within').drop(columns=['index_right', 'geometry'])
grid.drop(columns=['geometry'], inplace=True)
crash_severity.drop(columns=['geometry'], inplace=True)
merged_crash_severity.dropna(inplace=True)
merged_crash_severity['grid_index'] = merged_crash_severity['grid_index'].astype(int)

# set the non-numeric factors into dummy variables
crash_severity_dummies = pd.get_dummies(crash_severity, columns=['MAX_SEVERITY_LEVEL','ILLUMINATION', 'WEATHER1', 'ROAD_CONDITION', 'DVR_PRES_IND'])

spatial_factors_name = ['black_percentage', 'asian_percentage', 'hispanic_percentage', 'men_per_area', 'crime', 'nearest_police', 'investigation']
crash_factors_name = ['TRAVEL_SPD', 'DVR_PRES_IND', 'ILLUMINATION', 'WEATHER1', 'ROAD_CONDITION']
