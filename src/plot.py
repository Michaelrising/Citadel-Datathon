
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library based on matplotlib
import geopandas # working with geospatial data in python easier
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

color_set = ['#0c4e8c','#0c81e4','#11c4d4','#4fe7af','#ccf1cd',
           '#F9F3D1','#D4E537','#B3BE1A','#989400','#2C2B19',
           '#3A0E21','#EA0D62','#FD7B6E','#FDAD67','#FDDA64']

Traffic_path = '../APAC_2023_Datasets/Traffic, Investigations _ Other/'
Crash_path = '../APAC_2023_Datasets/Crashes/'

df_crash_info = pd.read_csv(Crash_path+'crash_info_general.csv')
df_crash_info = df_crash_info[df_crash_info.columns[1:]]
cond = (df_crash_info['DEC_LAT'].isna())|(df_crash_info['DEC_LONG'].isna())
df_crash_info = df_crash_info.loc[~cond,:]

df_crash_info = df_crash_info.rename(columns={'DEC_LAT':'lat','DEC_LONG':'lng'})

df_flag = pd.read_csv(Crash_path+'crash_info_flag_variables.csv')
df_flag = df_flag.merge(df_crash_info[['CRN','lat','lng','fips','CRASH_MONTH','CRASH_YEAR','DAY_OF_WEEK']],on=['CRN'])
df_police = pd.read_csv(Traffic_path+'police_stations.csv')

df_flag = gpd.GeoDataFrame(df_flag, geometry=gpd.points_from_xy(df_flag.lng, df_flag.lat))
df_flag = df_flag[df_flag.geometry.within(df_flag.geometry.unary_union.convex_hull)]
df_police = gpd.GeoDataFrame(df_police, geometry=gpd.points_from_xy(df_police.lng, df_police.lat))

crash_general = pd.read_csv('../APAC_2023_Datasets/Crashes/crash_info_general.csv')
crash_general = crash_general[crash_general['DEC_LAT'].notna() & crash_general['DEC_LONG'].notna()]
# crash_general['LATITUDE'] = crash_general['LATITUDE'].apply(convert_to_lat)
# crash_general['LONGITUDE'] = crash_general['LONGITUDE'].apply(convert_to_long)
crash_points = gpd.points_from_xy(crash_general.DEC_LONG, crash_general.DEC_LAT)
crash_gdf = gpd.GeoDataFrame(geometry=crash_points)

# count the car crashes in each census tract
census = gpd.read_file('../Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp')
crash_counts = gpd.sjoin(census, crash_gdf, op='contains').groupby(level=0).size().reset_index(name='count')
# if census tract does not contain any crash, set the count to 0
crash_counts = crash_counts.set_index('index').reindex(range(len(census))).fillna(0).reset_index()
fig, ax = plt.subplots(figsize=(10,8))
census.plot(column=crash_counts['count'], cmap='Blues', alpha=0.8, ax=ax)
# street = gpd.read_file('Data/CompleteStreets-shp/CompleteStreets.shp')
# fig, ax = plt.subplots( figsize=(10,10))
# street.plot(color=color_set[4], alpha=0.1, ax=ax)
# map the reported by police or not to color of the points
# df_flag['Reported by Police'] = df_flag.PSP_REPORTED.astype(int)
# df_flag['Reported by Police'] = df_flag['Reported by Police'].replace({0:'blue',1:'red'})
# p1 = df_flag[df_flag.CORE_NETWORK==0].plot(ax=plt.gca(), alpha=0.5, c=df_flag.loc[df_flag.CORE_NETWORK==0,'Reported by Police'],markersize=2)
# p2 = df_flag[(df_flag.FATAL==True)&(df_flag.SPEEDING==True)].plot(ax=plt.gca(), color='black', markersize=4, label='Fatal and Speeding Crashes')
# df_police.plot(ax=plt.gca(), color='black', markersize=50, marker='*')
df_flag[df_flag.UNSIGNALIZED_INT==1].plot(ax=plt.gca(), color=color_set[0], markersize=2) # harmany color?
# legend_elements = [Line2D([0], [0], marker='o', markerfacecolor='blue', color='w',label='Crashes Not Reported', markersize=5),
#                      Line2D([0], [0], marker='o', color='w',markerfacecolor='red', label='Crashes Reported', markersize=5),
#                    Line2D([0], [0], marker='*', color='w',markerfacecolor='black', label='Police Stations', markersize=10)]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
plt.title('Crashes at Un-signalized Intersections', fontsize=20)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('./figures/crashes_unsignalized.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
census.plot(column=crash_counts['count'], cmap='Blues', alpha=0.8, ax=ax)
# street = gpd.read_file('Data/CompleteStreets-shp/CompleteStreets.shp')
# fig, ax = plt.subplots( figsize=(10,10))
# street.plot(color=color_set[4], alpha=0.1, ax=ax)
# map the reported by police or not to color of the points
# df_flag['Reported by Police'] = df_flag.PSP_REPORTED.astype(int)
# df_flag['Reported by Police'] = df_flag['Reported by Police'].replace({0:'blue',1:'red'})
# p1 = df_flag[df_flag.CORE_NETWORK==0].plot(ax=plt.gca(), alpha=0.5, c=df_flag.loc[df_flag.CORE_NETWORK==0,'Reported by Police'],markersize=2)
# p2 = df_flag[(df_flag.FATAL==True)&(df_flag.SPEEDING==True)].plot(ax=plt.gca(), color='black', markersize=4, label='Fatal and Speeding Crashes')
# df_police.plot(ax=plt.gca(), color='black', markersize=50, marker='*')
df_flag[df_flag.AGGRESSIVE_DRIVING==1].plot(ax=plt.gca(), color=color_set[0], markersize=2) # harmany color?
# legend_elements = [Line2D([0], [0], marker='o', markerfacecolor='blue', color='w',label='Crashes Not Reported', markersize=5),
#                      Line2D([0], [0], marker='o', color='w',markerfacecolor='red', label='Crashes Reported', markersize=5),
#                    Line2D([0], [0], marker='*', color='w',markerfacecolor='black', label='Police Stations', markersize=10)]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
plt.title('Crashes with Aggressive Driving', fontsize=20)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('./figures/crashes_aggressive.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
census.plot(column=crash_counts['count'], cmap='Blues', alpha=0.8, ax=ax)
# street = gpd.read_file('Data/CompleteStreets-shp/CompleteStreets.shp')
# fig, ax = plt.subplots( figsize=(10,10))
# street.plot(color=color_set[4], alpha=0.1, ax=ax)
# map the reported by police or not to color of the points
# df_flag['Reported by Police'] = df_flag.PSP_REPORTED.astype(int)
# df_flag['Reported by Police'] = df_flag['Reported by Police'].replace({0:'blue',1:'red'})
# p1 = df_flag[df_flag.CORE_NETWORK==0].plot(ax=plt.gca(), alpha=0.5, c=df_flag.loc[df_flag.CORE_NETWORK==0,'Reported by Police'],markersize=2)
# p2 = df_flag[(df_flag.FATAL==True)&(df_flag.SPEEDING==True)].plot(ax=plt.gca(), color='black', markersize=4, label='Fatal and Speeding Crashes')
# df_police.plot(ax=plt.gca(), color='black', markersize=50, marker='*')
df_flag[(df_flag.SPEEDING==1) & (df_flag.SPEEDING_RELATED==1)].plot(ax=plt.gca(), color=color_set[0], markersize=2) # harmany color?

plt.title('Crashes with Speeding', fontsize=20)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('./figures/crashes_speeding.png', dpi=300)
plt.show()



fig, ax = plt.subplots(figsize=(10,5))
census.plot(column=crash_counts['count'], cmap='Blues', alpha=0.8, ax=ax)
# map the reported by police or not to color of the points
df_flag['Reported by Police'] = df_flag.PSP_REPORTED.astype(int)
df_flag['Reported by Police'] = df_flag['Reported by Police'].replace({0:color_set[0],1:color_set[-1]})
p1 = df_flag[df_flag.CORE_NETWORK==1].plot(ax=plt.gca(), alpha=0.5, c=df_flag.loc[df_flag.CORE_NETWORK==1,'Reported by Police'],markersize=2)
# lm = sns.lmplot(x='lng', y='lat',  hue='Reported by Police',
#            data=df_flag[(df_flag.lat>39.8)&(df_flag.lng<-75.0)&
#                         (df_flag.CORE_NETWORK==1)],
#            fit_reg=False, scatter_kws={'alpha':0.8,"s": 5}, palette='Blues')
df_flag[(df_flag.FATAL==True)&(df_flag.SPEEDING==True)].plot(ax=plt.gca(), color='black', markersize=4, label='Fatal and Speeding Crashes')
df_police.plot(ax=plt.gca(), color='red', markersize=50, marker='*')
# plt.scatter(x=-75.10, y=40.03, color='red', marker='*')
legend_elements = [Line2D([0], [0], color=color_set[0],label='Not Reported (Core Network)', markersize=5),
                     Line2D([0], [0], color=color_set[-1], label='Reported (Core Network)', markersize=5),
                   Line2D([0], [0], marker='*', color='w',markerfacecolor='red', label='Police Stations', markersize=10),
                   Line2D([0], [0], marker='o', color='w',markerfacecolor='black', label='Fatal and Speeding Crashes', markersize=5),
                   Line2D([0], [0], marker='*', color='w',markerfacecolor='black', label='ROOSEVELT BL S.', markersize=20)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, ) # no legend box
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('./figures/crashes_core_network.png', dpi=300)
plt.show()
