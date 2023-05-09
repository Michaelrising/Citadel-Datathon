import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


census = gpd.read_file('../Data/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp')
populations = pd.read_csv('../APAC_2023_Datasets/Traffic, Investigations _ Other/philadelphia_population_metrics.csv')
census['GEOID10'] = census['GEOID10'].astype(int)
populations = populations.merge(census[['GEOID10', 'geometry']], how='left', left_on='GEOGRAPHY_NAME', right_on='GEOID10')
populations = gpd.GeoDataFrame(populations)
fig6, ax = plt.subplots(figsize=(10,10))
populations.plot(column='PERCENT_ASIAN_NH', cmap='OrRd', legend=True, ax=ax)
ax.set_title('Population of black percentage')
plt.show()

