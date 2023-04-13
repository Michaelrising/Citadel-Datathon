# car crash data sets

crash_info_vehicle = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_vehicles.csv')
crash_info_roadway = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_roadway.csv')
crash_info_people = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_people.csv')

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


num_grid = len(grid)

severity_categories = len(severity_dummies.columns)

with pm.Model() as model:
    # Define hyperpriors for the upper-level spatial model (crash count)
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=10)
    sigma_alpha = pm.HalfCauchy("sigma_alpha", beta=5)

    # Define random effects for the census tracts
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=(num_tracts,))

    # Define priors for fixed effects (spatial model)
    spatial_beta = pm.Normal("spatial_beta", mu=0, sd=10, shape=(len(X_spatial.columns),))

    # Define spatial factors
    spatial_factors = merged_data[X_spatial.columns].values

    # Define the linear predictor for the crash count
    mu_count = alpha[merged_data["census_tract"].values] + pm.math.dot(spatial_factors, spatial_beta)

    # Choose either Poisson or Negative Binomial likelihood for the crash count
    # Poisson
    count_likelihood = pm.Poisson("count_likelihood", mu=np.exp(mu_count), observed=merged_data["Y_it"])

    # OR

    # Negative Binomial
    alpha_count = pm.Gamma("alpha_count", alpha=0.1, beta=0.1)
    count_likelihood = pm.NegativeBinomial("count_likelihood", mu=np.exp(mu_count), alpha=alpha_count,
                                           observed=merged_data["Y_it"])

    # Define priors for fixed effects (severity model)
    severity_beta = pm.Normal("severity_beta", mu=0, sd=10, shape=(len(X_crash.columns), severity_categories - 1))

    # Define crash-related factors
    crash_factors = merged_data[X_crash.columns].values

    # Define the linear predictor for the severity categories
    mu_severity = pm.math.dot(crash_factors, severity_beta)

    # Compute the probability of each severity category (softmax transformation)
    prob_severity = pm.math.softmax(pm.math.concatenate([mu_severity, np.zeros((mu_severity.shape[0], 1))], axis=1),
                                    axis=1)

    # Multinomial likelihood for the severity categories
    severity_likelihood = pm.Multinomial("severity_likelihood", n=1, p=prob_severity, observed=severity_dummies.values)


with model:
    trace = pm.sample(draws=2000, tune=1000, chains=4, cores=4)

pm.summary(trace).round(2)


