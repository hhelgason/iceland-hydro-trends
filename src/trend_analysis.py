import pandas as pd
import pandas as pds
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
from pathlib import Path
import geopandas as gpd
from statsmodels.tsa.stattools import acf
from statsmodels.graphics import tsaplots
import datetime as dt
import geopandas as gpds
import os
from scipy.stats import kendalltau
from pymannkendall import hamed_rao_modification_test
from scipy.stats import theilslopes
import seaborn as sns

# Define plot settings:

# Turn on Seaborn style
sns.set()
facecolor = 'white' #'lightgrey'
plt.rcParams['font.family'] = 'Arial' 
streamflow_markersize = 150
streamflow_sign_size = 18
map_fontsize = 20
map_fontsize_sea = 35

# Read the gauges shapefile that contains the indices and V numbers
gauges_gdf = gpd.read_file(Path(r"C:\Users\hordurbhe\Documents\Vinna\lamah\lamah_ice\lamah_ice\D_gauges\3_shapefiles\gauges.shp"))
gauges_gdf = gauges_gdf.set_index('id')
gauges_gdf = gauges_gdf.set_crs('epsg:3057')
gauges = gauges_gdf.copy()

# Read the catchment characteristics - Extract area_calc and human influence
catchments_chara = pds.read_csv(Path(r'C:\Users\hordurbhe\Documents\Vinna\lamah\lamah_ice\lamah_ice\A_basins_total_upstrm\1_attributes\Catchment_attributes.csv'),sep=';')
catchments_chara = catchments_chara.set_index('id')

# Read the catchment characteristics - Extract area_calc and human influence
hydro_sign = pds.read_csv(Path(r'C:\Users\hordurbhe\Documents\Vinna\lamah\lamah_ice\lamah_ice\D_gauges\1_attributes\hydro_indices_1981_2018_unfiltered.csv'),sep=';')
hydro_sign = hydro_sign.set_index('id')

# Read catchments
catchments = gpd.read_file(Path(r"C:/Users/hordurbhe/Documents/Vinna/lamah/lamah_ice/lamah_ice/A_basins_total_upstrm/3_shapefiles/Basins_A.shp"))
catchments = catchments.set_index('id')
catchments = catchments.set_crs('epsg:3057')

# Define plot specifications
colormap = 'RdBu'
seasonal_colnames = ['trend_DJF','trend_MAM', 'trend_JJA', 'trend_SON']
seasonal_pnames = ['pval_DJF','pval_MAM','pval_JJA', 'pval_SON']
seasonal_title_names = ['Dec-Feb','Mar-May','Jun-Aug','Sep-Nov']
shift_value = 100

vmin = -10
vmax = 10
annual_vmin = -6 #-2.5
annual_vmax = 6 #2.5
autocorr_vmin = -1
autocorr_vmax = 1

# For plotting:
def determine_extend(vmin,vmax,vmin_actual,vmax_actual):
    # Set 'extend' variable in accordance
    if vmin_actual < vmin and vmax_actual > vmax:
        extend = 'both'
    elif vmin_actual < vmin:
        extend = 'min'
    elif vmax_actual > vmax:
        extend = 'max'
    else:
        extend = 'neither'
    return(extend)

# Define the plot helper function
def plot_figs(basemap,glaciers,ax,iceland_shapefile_color,glaciers_color):
    # This function plots the basemap, the glaciers and sets x,ylimits and x,yticks
    # Set figure limits
    minx, miny = 222375, 307671
    maxx, maxy = 765246, 697520
    
    basemap.plot(ax=ax, color=iceland_shapefile_color,edgecolor='darkgray')
    glaciers.plot(ax=ax,facecolor=glaciers_color,edgecolor='none')
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def create_folders(base_folder,start_year,end_year):
    period = f"{start_year}_{end_year}"

    # Define folder paths
    base_path = base_folder / period
    daily_timeseries_path = base_path / 'daily_streamflow_series'
    #seasonal_trends_path = base_path / 'seasonal_trend_series'
    #monthly_trends_path = base_path / 'monthly_trend_series'
    #annual_trends_path = base_path / 'annual_trend_series'
    seasonal_trends_path_mod_ts = base_path / 'seasonal_trend_series_mod_ts'
    monthly_trends_path_mod_ts = base_path / 'monthly_trend_series_mod_ts'
    annual_trends_path_mod_ts = base_path / 'annual_trend_series_mod_ts'
    annual_autocorrelation_path = base_path / 'annual_autocorrelation'
    maps_path = base_path / 'maps'
    raster_trends_path = base_path / 'raster_trends'

    # Create folders
    for folder_path in [base_path, daily_timeseries_path, annual_autocorrelation_path, maps_path,raster_trends_path,seasonal_trends_path_mod_ts, annual_trends_path_mod_ts,monthly_trends_path_mod_ts]:
        folder_path.mkdir(parents=True, exist_ok=True)
    return(daily_timeseries_path, annual_autocorrelation_path, maps_path,raster_trends_path,seasonal_trends_path_mod_ts, annual_trends_path_mod_ts,monthly_trends_path_mod_ts)

def plot_streamflow_timeseries(df,gauges,path,year):
    for gauge_id in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df[gauge_id], label=f'Gauge {gauge_id}')
        plt.title('Gauge %s, %s %s - %s' % (gauge_id,gauges.loc[int(gauge_id)]['river'], gauges.loc[int(gauge_id)]['name'],year),)
        plt.xlabel('Date')
        plt.ylabel('Streamflow')
        plt.legend()
        plt.grid()
        savename = '%s.png' % gauge_id
        save_path = os.path.join(path,savename)
        plt.savefig(save_path,dpi=300)

def return_df(df,year,missing_data_threshold=0.8):
    # df: Contains streamflow data
    # Filter columns (gauges) containing data for the given year
    year_data = df['%s-10-01' % (int(year)-1):'2021-09-30'].copy()
    gauge_ids_to_export = []
    for gauge_id in year_data.columns:
        to_plot = year_data[gauge_id]
        if len(to_plot.dropna()) / len(pds.date_range('%s-10-01' % year,'2021-09-30'))>missing_data_threshold:
            gauge_ids_to_export.append(gauge_id)
        else:
            print('Gauge omitted: Overall raw coverage between %s and 2021 is less than %s for gauge id %s (%s, %s)' % (year, missing_data_threshold,gauge_id,gauges.loc[int(gauge_id)]['river'],gauges.loc[int(gauge_id)]['name']))
    return(year_data[gauge_ids_to_export])
    
def return_df_updated(df,start_year,end_year,missing_data_threshold=0.8):
    # df: Contains streamflow data
    # Filter columns (gauges) containing data for the given year
    year_data = df['%s-10-01' % (int(start_year)):'%s-09-30'% (int(end_year))].copy() 
    gauge_ids_to_export = []
    for gauge_id in year_data.columns:
        to_plot = year_data[gauge_id]
        if len(to_plot.dropna()) / len(pds.date_range('%s-10-01' % start_year,'%s-09-30' % end_year))>missing_data_threshold:
            gauge_ids_to_export.append(gauge_id)
        else:
            print('Gauge omitted: Overall raw coverage between %s and %s is less than %s for gauge id %s (%s, %s)' % (start_year, end_year, missing_data_threshold,gauge_id,gauges.loc[int(gauge_id)]['river'],gauges.loc[int(gauge_id)]['name']))
    return(year_data[gauge_ids_to_export])
    
def return_df_extended(df,year,missing_data_threshold=0.8):
    # This function is like return_df but returns also the years after 2021 (if there are any)
    # df: Contains streamflow data
    # Filter columns (gauges) containing data for the given year
    year_data = df['%s-10-01' % (int(year)-1):'2024-09-30'].copy() # ['%s-09-30' % int(year)-1:].copy()
    gauge_ids_to_export = []
    for gauge_id in year_data.columns:
        to_plot = year_data[gauge_id]
        if len(to_plot.dropna()) / len(pds.date_range('%s-10-01' % year,'2021-09-30'))>missing_data_threshold:
            gauge_ids_to_export.append(gauge_id)
        else:
            print('Gauge omitted: Overall raw coverage between %s and 2021 is less than %s for gauge id %s (%s, %s)' % (year, missing_data_threshold,gauge_id,gauges.loc[int(gauge_id)]['river'],gauges.loc[int(gauge_id)]['name']))
    return(df[gauge_ids_to_export])

def calc_trend_and_pval(data):
    tmul=10
    # Calculate the linear trend using numpy's polyfit function
    trend, intercept = np.polyfit(range(len(data)), data, 1)
    trend_ts, intercept_ts, _, _ = theilslopes(data, range(len(data)))

    # Calculate the p-value using the Mann-Kendall test
    _, pval = kendalltau(data, range(len(data)))
    _, _, mod_pval, _, _, _, _,_,_ = hamed_rao_modification_test(data)

    # Convert the trends to percent increase per decade
    trend_percent_increase_per_decade = (trend / np.mean(data)) * 100 * tmul
    trend_percent_increase_per_decade_ts = (trend_ts / np.mean(data)) * 100 * tmul

    # Return additional variables for plotting
    x_values = np.arange(len(data))

    return trend_percent_increase_per_decade, pval, trend, intercept, x_values,trend_percent_increase_per_decade_ts,mod_pval,trend_ts,intercept_ts, trend_ts

def calc_trends(df, gauges, missing_data_threshold=0.8):
    # Calculate annual (water years), monthly, and seasonal averages for each column
    water_years = [(d - dt.timedelta(days=273)).year for d in df.index]
    annual_avg = df.groupby(water_years).mean()
    annual_avg.index = pds.to_datetime(['%s-12-31' %i for i in df.groupby(water_years).mean().index]) 
    monthly_avg = df.resample('M').mean()
    seasonal_avg = df.resample('QS-DEC').mean()

    annual_count = df.groupby(water_years).count()
    annual_count.index = pds.to_datetime(['%s-12-31' %i for i in df.groupby(water_years).mean().index]) 
    monthly_count = df.resample('M').count()
    seasonal_count = df.resample('QS-DEC').count()

    valid_data_dict = dict()
    invalid_data_dict = dict()
    plot_dict = dict()
    plot_dict_mod = dict()

    # Calculate trends and significance levels for each column in each dataframe
    results = pd.DataFrame(columns=['annual_trend', 'pval'])
    for col in df.columns:
        # Annual trends
        valid = annual_avg[col][(annual_count[col].dropna()/365)>0.9]
        valid_data_percent = len(valid)/len(annual_avg[col])
        annual_data = valid.dropna()
        if valid_data_percent >= missing_data_threshold:
            annual_trend, annual_pval, trend, intercept, x_values, annual_trend_ts,mod_pval,slope_ts,intercept_ts,trend_ts = calc_trend_and_pval(annual_data.values)
            results.loc[col, 'annual_trend'] = annual_trend
            results.loc[col, 'annual_trend_ts'] = annual_trend_ts
            results.loc[col, 'pval'] = annual_pval
            results.loc[col, 'pval_mod'] = mod_pval
            valid_data_dict[col,'annual'] = annual_data
            plot_dict[col,'annual'] = (trend, intercept, x_values)
            plot_dict_mod[col,'annual'] = (slope_ts, intercept_ts, x_values)

            # Low flows (10th percentile)
            low_flow_data = df[col].groupby(water_years).apply(lambda x: np.percentile(x, 10))
            _, _, _, _, low_flow_x_values,low_flow_trend,low_flow_pval,low_flow_slope,low_flow_intercept,trend_ts = calc_trend_and_pval(low_flow_data.dropna().values)
            high_flow_data = df[col].groupby(water_years).apply(lambda x: np.percentile(x, 90))
            _, _, _, _, high_flow_x_values,high_flow_trend,high_flow_pval,high_flow_slope,high_flow_intercept,trend_ts = calc_trend_and_pval(high_flow_data.dropna().values)

            results.loc[col, 'low_flow_trend'] = low_flow_trend
            results.loc[col, 'low_flow_pval'] = low_flow_pval
            results.loc[col, 'high_flow_trend'] = high_flow_trend
            results.loc[col, 'high_flow_pval'] = high_flow_pval
            valid_data_dict[col, 'low_flow'] = low_flow_data
            valid_data_dict[col, 'high_flow'] = high_flow_data
            plot_dict_mod[col, 'low_flow'] = (low_flow_slope, low_flow_intercept, low_flow_x_values)
            plot_dict_mod[col, 'high_flow'] = (high_flow_slope, high_flow_intercept, high_flow_x_values)

        else:
            invalid_data_dict[col,'annual'] = annual_data

        # JAS trends
        jas_data = df[col][df.index.month.isin([7, 8, 9])].resample('A-SEP').mean()
        jas_count = df[col][df.index.month.isin([7, 8, 9])].resample('A-SEP').count()
        valid = jas_data[(jas_count / 90) > missing_data_threshold]
        valid_data_percent = len(valid) / len(jas_data)
        jas_data = valid.dropna()
        if valid_data_percent >= missing_data_threshold:
            jas_trend, jas_pval, trend, intercept, x_values, jas_trend_ts, mod_pval, slope_ts, intercept_ts, trend_ts = calc_trend_and_pval(jas_data.values)
            results.loc[col, 'trend_JAS'] = jas_trend
            results.loc[col, 'pval_JAS'] = jas_pval
            results.loc[col, 'trend_JAS_ts'] = jas_trend_ts
            results.loc[col, 'pval_JAS_mod'] = mod_pval
            valid_data_dict[col, 'JAS'] = jas_data
            plot_dict[col, 'JAS'] = (trend, intercept, x_values)
            plot_dict_mod[col, 'JAS'] = (slope_ts, intercept_ts, x_values)
        else:
            invalid_data_dict[col, 'JAS'] = jas_data
            print(f"Gauge omission: For gauge id {col}, valid data percent is {valid_data_percent:.2f} for JAS ({gauges.loc[int(col)]['river']}, {gauges.loc[int(col)]['name']})")

        
        # Seasonal trends
        for month, season in zip([12, 3, 6, 9], ['DJF', 'MAM', 'JJA', 'SON']):
            # "valid" is a dataframe containing only the seasonal averages 
            # where more than X% data is available (X is the missing_data_threshold)
            valid = seasonal_avg[col][(seasonal_count[col].dropna()/90)>missing_data_threshold]
            # "valid_data_percent" is the number of valid individual seasons (e.g. number of valid winter seasons) 
            # divided by the total number of this particular season (winter, summer, spring, fall) in the series
            valid_data_percent = len(valid[valid.index.month==month])/len(seasonal_avg[seasonal_avg.index.month == month][col]) 

            seasonal_data = valid[valid.index.month==month].dropna() 
            if valid_data_percent >= missing_data_threshold:
                seasonal_trend, seasonal_pval, trend, intercept, x_values, seasonal_trend_ts,seasonal_mod_pval,slope_ts,intercept_ts,trend_ts = calc_trend_and_pval(seasonal_data.values)
                results.loc[col, f'trend_{season}'] = seasonal_trend
                results.loc[col, f'trend_{season}_ts'] = seasonal_trend_ts
                results.loc[col, f'pval_{season}'] = seasonal_pval
                results.loc[col, f'pval_{season}_mod'] = seasonal_mod_pval
                valid_data_dict[col,season] = seasonal_data
                plot_dict[col,season] = (trend, intercept, x_values)
                plot_dict_mod[col,season] = (slope_ts, intercept_ts, x_values)
            else:
                invalid_data_dict[col,season] = seasonal_data
                print('Gauge omission: For gauge id %s, valid data percent is %s for season %s (%s, %s)' %(col,valid_data_percent,season,gauges.loc[int(col)]['river'],gauges.loc[int(col)]['name']))

        # Monthly trends
        for month in np.arange(1,13,1):
            valid = monthly_avg[col][(monthly_count[col].dropna()/30)>missing_data_threshold]
            valid_data_percent = len(valid[valid.index.month==month])/len(monthly_avg[monthly_avg.index.month == month][col]) 
            monthly_data = valid[valid.index.month==month].dropna() 
            if valid_data_percent >= missing_data_threshold:
                monthly_trend, monthly_pval, trend, intercept, x_values,monthly_trend_ts,mod_pval,slope_ts,intercept_ts,trend_ts = calc_trend_and_pval(monthly_data.values)
                results.loc[col, f'trend_{month}'] = monthly_trend
                results.loc[col, f'pval_{month}'] = monthly_pval
                results.loc[col, f'trend_{month}_ts'] = monthly_trend_ts
                results.loc[col, f'pval_{month}_mod'] = mod_pval
                valid_data_dict[col,month] = monthly_data
                plot_dict[col,month] = (trend, intercept, x_values)
                plot_dict_mod[col,month] = (slope_ts, intercept_ts, x_values)
            else:
                invalid_data_dict[col,month] = monthly_data
                print('Gauge omission: For gauge id %s, valid data percent is %s for month %s (%s, %s)' %(col,valid_data_percent,month,gauges.loc[int(col)]['river'],gauges.loc[int(col)]['name']))        

    return results,valid_data_dict,invalid_data_dict,plot_dict,plot_dict_mod
    
def plot_raster_trends_with_significance(df, start_year, end_year, missing_data_threshold, raster_trends_path, sortby, variable, window_length):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import kendalltau
    sns.reset_orig()
    fontsize = 15
    labelpad = 2
    
    unit_dict = {'streamflow':'%',
    'prec': '%',
    'total_et': 'mm',
    '2m_temp_mean': '°C',
    'swe': '%',
    'surf_net_therm_rad_mean': 'Wm-2',
    'surf_net_solar_rad_mean': 'Wm-2',
    'prec_carra': '%%',
    'prec_rav': '%',
    'total_et_rav': 'mm',
    '2m_temp_rav': '°C',
    'surf_dwn_therm_rad_rav': 'Wm-2',
    'surf_outg_therm_rad_rav': 'Wm-2',
    'surf_net_therm_rad_rav': 'Wm-2'
     }

    colorbar_titles_dict = {'streamflow': 'streamflow',
    'prec': 'precipitation',
    'total_et': 'evapotranspiration',
    '2m_temp_mean': '2m temperature',
    'swe': 'Max SWE',
    'surf_net_therm_rad_mean': 'net thermal radiation (ERA5-Land - Downwards flux is positive)',
    'surf_net_solar_rad_mean': 'net solar radiation (ERA5-Land - Downwards flux is positive)',
    'prec_carra': 'precipitation (CARRA)',
    'prec_rav': 'precipitation (RAV-II)',
    'total_et_rav': 'evapotranspiration (RAV-II)',
    '2m_temp_rav': '2m temperature (RAV-II)',
    'surf_dwn_therm_rad_rav': 'downwelling thermal radiation (RAV-II)',
    'surf_outg_therm_rad_rav': 'outgoing thermal radiation (RAV-II)',
    'surf_net_therm_rad_rav': 'net thermal radiation (RAV-II - Downwards flux is positive)'}

    percent_var_list = ['streamflow','prec','prec_rav','prec_carra','swe','surf_net_therm_rad_mean','surf_net_solar_rad_mean','surf_dwn_therm_rad_rav',
                        'surf_outg_therm_rad_rav', 'surf_net_therm_rad_rav']

    min_year_count = (end_year - start_year) * missing_data_threshold
    print('min year count %s ' % min_year_count)
    df_heat = df.copy()
    df_heat = df_heat['%s-01-01' % start_year:'%s-12-31' % end_year]

    # Step 1: Calculate the X-day moving averages
    df_ma = df_heat.rolling(window=window_length,center=True).mean()

    # Step 2: Calculate the trend and Mann-Kendall p-values for each day of the year and each gauge
    trends = []
    p_values = []
    
    series_dict = dict()
    
    for day in range(1, 366):
        trend_per_gauge = []
        p_values_per_gauge = []

        for column in df_ma.columns:
            daily_values = df_ma[df_ma.index.dayofyear == day][column]
            
            # Ensure there are at least 75% non-null values for fitting a trend
            if daily_values.count() >= min_year_count:
                x = np.arange(len(daily_values.dropna()))
                trend = np.polyfit(x, daily_values.dropna().values, 1)[0] * 10
                if variable in percent_var_list:
                    trend = 100 * trend / daily_values.mean()
                trend_per_gauge.append(trend)
                series_dict[column,day] = daily_values

                # Calculate Mann-Kendall p-value
                _, p_value = kendalltau(np.arange(len(daily_values.dropna())), daily_values.dropna())
                p_values_per_gauge.append(p_value)
            else:
                trend_per_gauge.append(np.nan)
                p_values_per_gauge.append(np.nan)

        trends.append(trend_per_gauge)
        p_values.append(p_values_per_gauge)

    # Step 3: Create a DataFrame for the trends and p-values
    trend_df_percent = pd.DataFrame(trends, columns=df_ma.columns, index=range(1, 366))
    p_values_df = pd.DataFrame(p_values, columns=df_ma.columns, index=range(1, 366))

    if variable == 'streamflow':
        listi = catchments_chara[sortby].loc[[int(i) for i in trend_df_percent.columns]].sort_values().index.tolist()
        listi_str = [str(i) for i in listi]
        trend_df_percent_glac_sort = trend_df_percent[listi_str]
        p_values_df = p_values_df[listi_str]
    else:
        trend_df_percent_glac_sort = trend_df_percent

    # Drop all-nan rows
    trend_df_percent_glac_sort = trend_df_percent_glac_sort.dropna(axis=1, how='all')
    p_copy = p_values_df.copy()
    p_values_df = p_values_df.dropna(axis=1, how='all')

    # Plot the heatmap with transparency based on significance
    plt.figure(figsize=(12, 8))
    vmin_actual = trend_df_percent_glac_sort.min().min()
    vmax_actual = trend_df_percent_glac_sort.max().max()

    # Plot the heatmap without smoothing
    if variable in ['streamflow']: 
        vmin=-18
        vmax=18
        cmap = sns.diverging_palette(220, 20, as_cmap=True).reversed()
        cmap.set_bad(color='#404040')
    elif variable in ['2m_temp_mean','2m_temp_rav']:
        vmin=-1.8
        vmax=1.8
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        cmap.set_bad(color='#404040')
    elif variable in ['swe', 'surf_dwn_therm_rad_rav', 'surf_outg_therm_rad_rav']:
        vmax = max(abs(vmin_actual), abs(vmax_actual))
        vmin = -vmax
        cmap = sns.diverging_palette(220, 20, as_cmap=True).reversed()
    elif variable in ['total_et','total_et_rav','prec','prec_rav','prec_carra']:
        vmax = 20
        vmin = -vmax
        cmap = sns.diverging_palette(220, 20, as_cmap=True).reversed()
    elif variable in ['surf_net_solar_rad_mean','surf_net_therm_rad_mean','surf_net_therm_rad_rav']:
        vmax = 20
        vmin = -vmax
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
    else:
        print('not found')
    extend = determine_extend(vmin,vmax,vmin_actual,vmax_actual)
    im = plt.imshow(trend_df_percent_glac_sort.values.T, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')

    # Manually calculate and set y-tick positions in the middle of the rows
    ytick_positions = np.arange(0, len(trend_df_percent_glac_sort.columns))
    ytick_labels = trend_df_percent_glac_sort.columns
    plt.yticks(ytick_positions, ytick_labels)

    # Plot column-specific contours for significant trends (p-value < 0.05)
    for column_index, column_name in enumerate(p_values_df.columns):
        
        p_values = p_values_df[column_name]

        # Mask non-significant trends with hatch pattern
        mask = p_values < 0.05  # Areas with p-value larger than 0.05 are considered insignificant
        insignificant_indices = np.where(mask)[0]

        if len(insignificant_indices) > 0:
            # Identify continuous periods with insignificant trends
            insignificant_periods = []
            current_period = [insignificant_indices[0]]

            for i in range(1, len(insignificant_indices)):
                if insignificant_indices[i] == insignificant_indices[i-1] + 1:
                    current_period.append(insignificant_indices[i])
                else:
                    insignificant_periods.append(current_period)
                    current_period = [insignificant_indices[i]]

            insignificant_periods.append(current_period)

            # Plot contours for each insignificant period
            for period in insignificant_periods:
                start_x = period[0] - 0.5
                end_x = period[-1] + 0.5
                plt.hlines(column_index, start_x, end_x, colors='black', linewidths=0.5, linestyles='dashed')

    if variable in percent_var_list:
        cbar = plt.colorbar(im, label='Trend (% per decade)',extend=extend)
        cbar.set_label('Trend (% per decade)',fontsize=fontsize,labelpad=labelpad)
    elif variable in ['2m_temp_mean','2m_temp_rav']:
        cbar = plt.colorbar(im, label='Trend (°C per decade)',extend=extend)
        cbar.set_label('Trend (°C per decade)',fontsize=fontsize,labelpad=labelpad)
    elif variable in ['total_et','total_et_rav']:
        cbar=plt.colorbar(im, label='Trend (mm per decade)',extend=extend)
        cbar.set_label('Trend (mm per decade)',fontsize=fontsize,labelpad=labelpad)
    else:
        plt.colorbar(im, label='Trend (X per decade)',extend=extend)

    cbar.ax.tick_params(labelsize=fontsize)
    if start_year == 1973:
        plt.title('a) Sub-seasonal trend in %s (%s-%s)' % (colorbar_titles_dict[variable],start_year, end_year),fontsize=fontsize)
    elif start_year == 1993:
        plt.title('b) Sub-seasonal trend in %s (%s-%s)' % (colorbar_titles_dict[variable],start_year, end_year),fontsize=fontsize)
    else:
        plt.title('Sub-seasonal trend in %s (%s-%s)' % (colorbar_titles_dict[variable],start_year, end_year),fontsize=fontsize)
    plt.xlabel('Month',fontsize=fontsize)
    plt.ylabel('Streamflow Gauge ID',fontsize=fontsize,labelpad=labelpad)

    # Set x-axis ticks to be at the first day of each month
    months_ticks = np.arange(0, 335, 30.4)
    plt.xticks(months_ticks, df_ma.resample('M').mean().index.strftime('%b')[months_ticks.astype(int) // 30],fontsize=fontsize)
    ytick_positions = np.arange(0, len(trend_df_percent_glac_sort.columns))
    ytick_labels = trend_df_percent_glac_sort.columns
    plt.yticks(ytick_positions, ytick_labels,fontsize=fontsize)
    
    # Create a secondary y-axis
    ax=plt.gca()
    ax2 = ax.twinx()

    # Set the position of the secondary y-axis
    ax2.set_ylim(ax.get_ylim())

    # Get the % glaciation values sorted in the same order as your y-tick labels
    glaciation_values = [int(i) for i in 100 * catchments_chara.loc[[int(i) for i in trend_df_percent_glac_sort.dropna(axis=1, how='all').columns]]['g_frac'].sort_values()]

    # Set the labels for the secondary y-axis
    ax2.set_yticks(ytick_positions)
    ax2.set_yticklabels(glaciation_values,fontsize=fontsize)

    # Set the label for the secondary y-axis
    ax2.set_ylabel('% Glaciation', fontsize=fontsize,labelpad=-2)
    
    plt.tight_layout()
    
    save_path = os.path.join(raster_trends_path,'%sdMA_trends_%s.png' % (window_length,variable))
    plt.tight_layout()
    
    # Extract the directory from the save path
    directory = os.path.dirname(save_path)
    
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close()
    
    return p_values_df, p_copy, trend_df_percent,start_x,end_x,insignificant_indices,trend_df_percent_glac_sort,series_dict,df_ma

def plot_trendfigs(catchments, which_plots, merged_gdf, start_year, end_year, results,valid_data_dict,invalid_data_dict,plot_dict,plot_dict_mod, daily_timeseries_path, annual_autocorrelation_path, maps_path,raster_trends_path,seasonal_trends_path_mod_ts, annual_trends_path_mod_ts,monthly_trends_path_mod_ts):
    # Define plot specifications
    colormap = 'RdBu'
    
    # Trend per decade?
    tmul = 10

    # Specify some plot attributes
    iceland_shapefile_color = 'gray'
    glaciers_color = 'white'

    # Specify where to save the figures
    savepath = Path(r'C:\Users\hordurbhe\OneDrive - Landsvirkjun\Changes in streamflow in Iceland\figures')

    # Define the date string for plot savenames
    today = dt.date.today()
    today_str = '%s-%s-%s' % (today.year,today.month,today.day)

    # Read basemap for Iceland
    bmap = gpds.read_file(r'C:\Users\hordurbhe\Documents\Vinna\lamah\lamah_ice\stanford-xz811fy7881-shapefile\island_isn93.shp')

    # Read glacier outlines
    gpath = Path(r'C:\Users\hordurbhe\Documents\Vinna\lamah\lamah_ice\glacier_outline_1890_2019_hh_Aug2021\jökla-útlínur\2019_glacier_outlines.shp')
    glaciers = gpds.read_file(gpath)
    
    ############################### Plot the annual trends on a map - MOD and Thiel-Sen ###############################################
    if which_plots['annual_map']:
        # Create a figure with a single subplot
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(facecolor)
  
        # Calculate the min and max of the values being plotted
        data_min = merged_gdf['annual_trend_ts'].min()
        data_max = merged_gdf['annual_trend_ts'].max()
        extend = determine_extend(annual_vmin,annual_vmax,data_min,data_max)
        # Plot the map
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
    
        im = merged_gdf.plot(column='annual_trend_ts', legend=False, vmin=annual_vmin, vmax=annual_vmax,
                             legend_kwds={'label': 'Trend in streamflow from %s-%s (%%/decade)' % (start_year, end_year),
                                          'orientation': "horizontal",
                                          'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap,s=streamflow_markersize)
    
        significant_points = merged_gdf[merged_gdf['pval_mod'] < 0.05]
        ax.plot(significant_points.geometry.x-shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markerfacecolor='none', markeredgecolor='k',linestyle='none',lw='0.5')
        
        minx, miny = 222375, 307671
        maxx, maxy = 765246, 697520
        ax.set_xlim(minx,maxx)
        ax.set_ylim(miny,maxy)
        
        # Create a ScalarMappable to connect the color map to the color bar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax))
        sm.set_array([])  # dummy array
        ax.set_title('Annual streamflow',size=map_fontsize,y=0.95)
        catchments.loc[merged_gdf['annual_trend_ts'].dropna().index].plot(facecolor='none', edgecolor='black',ax=ax,zorder=3,lw=0.25)

        # Add colorbar below the plot
        cax = fig.add_axes([0.25, 0.07, 0.6, 0.03])  # Move the colorbar a little to the right        
        cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax), orientation='horizontal',extend=extend)
        cb.set_label('Trend in streamflow from %s-%s (%%/decade)' % (start_year, end_year),size=map_fontsize)
        cb.ax.tick_params(labelsize=map_fontsize)  # Adjust font size if needed
        save_path = os.path.join(maps_path,'annual_trend_mod_ts.png')
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
        plt.close()

    ################################### Plot the seasonal trends on a map - MOD and Thiel-Sen ################################################
    if which_plots['seasonal_map']:
    
        # Calculate the min and max of the values being plotted
        columns = ['trend_DJF', 'trend_MAM', 'trend_JJA', 'trend_SON']
        data_min = merged_gdf[columns].min().min()  # Find the overall minimum
        data_max = merged_gdf[columns].max().max()  # Find the overall maximum
        extend = determine_extend(annual_vmin,annual_vmax,data_min,data_max)
        # define subplot grid
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor(facecolor)
        plt.subplots_adjust(hspace=-0, wspace=0)
    
        i=0
        for col, ax in zip(seasonal_colnames, axs.ravel()):
            plot_figs(bmap,glaciers,ax,iceland_shapefile_color,glaciers_color)
            mask = catchments_chara['degimpact'] != 's'
            im=merged_gdf.loc[mask].plot(column=col+'_ts',legend=False, vmin=vmin,vmax=vmax,
                                legend_kwds={'label': 'Trend in streamflow from %s-%s (%%/decade)' %(start_year,end_year),
                                         'orientation': "horizontal",
                                         'shrink': 0.8, 'pad':0.03}, ax=ax, cmap=colormap, s=200, zorder=2) 
            # filter dataframe to get significant points
            significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask][seasonal_pnames[i]+'_mod'] < 0.05]
            ax.set_title(seasonal_title_names[i],y=0.9,fontsize=map_fontsize_sea)
            title = ax.title
            catchments.loc[merged_gdf[col].loc[mask].dropna().index].plot(facecolor='none', edgecolor='black',ax=ax,zorder=3,lw=0.25)
            ax.plot(significant_points.geometry.x-shift_value, significant_points.geometry.y, marker='o', markersize=21, markerfacecolor='none', markeredgecolor='k',linestyle='none',lw='0.5',zorder=1)
    
            i+=1
    
        # create scalar mappable for colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin,vmax=vmax)) 
        sm._A = [] # set empty array for now, will be set later by imshow
    
        # add common colorbar below subplots
        cb=fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.05, 
                     label='Trend in streamflow from %s-%s (%%/decade)' %(start_year,end_year),shrink=0.6,extend=extend)
        cb.set_label('Trend in streamflow from %s-%s (%%/decade)' % (start_year, end_year),size=map_fontsize_sea)

        cb.ax.tick_params(labelsize=map_fontsize_sea) # set font size of colorbar label
        cb.ax.xaxis.label.set_size(map_fontsize_sea)
        cb.ax.tick_params(labelsize=map_fontsize_sea)

        save_path = os.path.join(maps_path,'seasonal_trend_ts_mod.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
            
            
    if which_plots['glaciated_basins']:
        # Calculate the min and max of the values being plotted
        columns = ['trend_JAS_ts', 'annual_trend_ts']
        data_min = merged_gdf[columns].where(merged_gdf['g_frac']>0.05).min().min()  # Find the overall minimum
        data_max = merged_gdf[columns].where(merged_gdf['g_frac']>0.05).max().max()  # Find the overall maximum
        extend = determine_extend(annual_vmin,annual_vmax,data_min,data_max)
    
        # Get the global min and max values
        global_vmin = data_min
        global_vmax = data_max
        
        global_vmax = max(abs(global_vmin), abs(global_vmax))
        global_vmin = -global_vmax
     
        # Create a figure with two subplots (one row, two columns)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        fig.patch.set_facecolor(facecolor)
        
        # Plot trends for annual average flow 
        ax = axs[0]
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = catchments_chara['g_frac'] > 0.05
        im = merged_gdf.loc[mask].plot(column='annual_trend_ts', legend=False, vmin=global_vmin, vmax=global_vmax,
                                       legend_kwds={'label': 'Trend in 10th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                                    'orientation': "horizontal",
                                                    'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap, s=streamflow_markersize, zorder=2)
        significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['pval_mod'] < 0.05]
        ax.plot(significant_points.geometry.x - shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markeredgewidth =2,
                markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5', zorder=1)
        if start_year == 1973:
            ax.set_title('a) Trend in annual average streamflow in glaciated basins\n1973-2023', size=map_fontsize, y=0.95)
        elif start_year == 1993:
            ax.set_title('c) Trend in annual average streamflow in glaciated basins\n1993-2023', size=map_fontsize, y=0.95)
        else:
            ax.set_title('Trend in annual average streamflow in glaciated basins', size=20, y=0.95)
        
        catchments.loc[merged_gdf['annual_trend_ts'].loc[mask].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        
        # Plot high flow trends
        ax = axs[1]
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = (catchments_chara['degimpact'] != 's') & (catchments_chara['g_frac'] > 0.05)
        im = merged_gdf.loc[mask].plot(column='trend_JAS_ts', legend=False, vmin=global_vmin, vmax=global_vmax,
                                       legend_kwds={'label': 'Trend in 90th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                                    'orientation': "horizontal",
                                                    'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap, s=streamflow_markersize, zorder=2)
        significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['pval_JAS_mod'] < 0.05]
        ax.plot(significant_points.geometry.x - shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markeredgewidth =2,
                markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5', zorder=1)
        if start_year == 1973:
            ax.set_title('b) Trend in JAS streamflow in glaciated basins\n1973-2023', size=map_fontsize, y=0.95)
        elif start_year == 1993:
            ax.set_title('d) Trend in JAS streamflow in glaciated basins\n1993-2023', size=map_fontsize, y=0.95)
        else:
            ax.set_title('Trend in JAS streamflow in glaciated basins', size=map_fontsize, y=0.95)
        catchments.loc[merged_gdf['trend_JAS_ts'].loc[mask].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
    
        # Add a common colorbar below the subplots
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=global_vmin, vmax=global_vmax))
        sm.set_array([])  # Dummy array
        cax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
        cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=global_vmin, vmax=global_vmax), orientation='horizontal')
        cb.set_label('Trend in streamflow %s-%s [%%/decade]' % (start_year, end_year), size=map_fontsize)
        cb.ax.tick_params(labelsize=map_fontsize)
        
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(maps_path, 'glaciated_basins_flow_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        save_path = os.path.join(maps_path, 'glaciated_basins_flow_trends.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    ############################### Plot the low flow trends on a map ###############################################
    if which_plots['low_flow_map']:
        # Create a figure with a single subplot
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(facecolor)

        # Plot your map
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = catchments_chara['degimpact'] != 's'
        im = merged_gdf.loc[mask].plot(column='low_flow_trend', legend=False, vmin=annual_vmin, vmax=annual_vmax,
                             legend_kwds={'label': 'Trend in 10th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                          'orientation': "horizontal",
                                          'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap, s=streamflow_markersize,zorder=2)
    
        significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['low_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x-shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markerfacecolor='none', markeredgecolor='k',linestyle='none',lw='0.5',zorder=1)
        # Create a ScalarMappable to connect the color map to the color bar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax))
        sm.set_array([])  # dummy array
        ax.set_title('Annual low flow (10th percentile)',size=14,y=0.95)
        catchments.loc[merged_gdf['annual_trend_ts'].dropna().index].plot(facecolor='none', edgecolor='black',ax=ax,zorder=3,lw=0.25)

        # Add colorbar below the plot
        cax = fig.add_axes([0.25, 0.07, 0.6, 0.03])  # Move the colorbar a little to the right
    
        cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax), orientation='horizontal')
        cb.set_label('Trend in low flow from %s-%s [%%/decade]' % (start_year, end_year),size=14)
        cb.ax.tick_params(labelsize=12)  # Adjust font size if needed
        save_path = os.path.join(maps_path,'low_flow_trends.png')
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
        plt.close()
    
    ############################### Plot the high flow trends on a map ###############################################
    if which_plots['high_flow_map']:
        # Create a figure with a single subplot
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(facecolor)

        # Plot your map
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = catchments_chara['degimpact'] != 's'
        im = merged_gdf.loc[mask].plot(column='high_flow_trend', legend=False, vmin=annual_vmin, vmax=annual_vmax,
                             legend_kwds={'label': 'Trend in 90th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                          'orientation': "horizontal",
                                          'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap,s=streamflow_markersize,zorder=2)
    
        significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['high_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x-shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markerfacecolor='none', markeredgecolor='k',linestyle='none',lw='0.5',zorder=1)
        # Create a ScalarMappable to connect the color map to the color bar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax))
        sm.set_array([])  # dummy array
        ax.set_title('Annual high flow (90th percentile)',size=14,y=0.95)
        catchments.loc[merged_gdf['annual_trend_ts'].dropna().index].plot(facecolor='none', edgecolor='black',ax=ax,zorder=3,lw=0.25)

        # Add colorbar below the plot
        cax = fig.add_axes([0.25, 0.07, 0.6, 0.03])  # Move the colorbar a little to the right
    
        cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=annual_vmin, vmax=annual_vmax), orientation='horizontal')
        cb.set_label('Trend in high flow from %s-%s [%%/decade]' % (start_year, end_year),size=14)
        cb.ax.tick_params(labelsize=12)  # Adjust font size if needed
        save_path = os.path.join(maps_path,'high_flow_trends.png')
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
        plt.close()
                
                
    # Plot high and low flows on a map, 1x2 subplot:
    if which_plots['low_flow_map']:
    
    # Calculate global vmin and vmax from both low flow and high flow trends
        low_flow_values = merged_gdf['low_flow_trend'].loc[mask].drop([15,48], errors='ignore').dropna()
        high_flow_values = merged_gdf['high_flow_trend'].loc[mask].drop([15,48], errors='ignore').dropna()
        
        # Get the global min and max values
        global_vmin = min(low_flow_values.min(), high_flow_values.min())
        global_vmax = max(low_flow_values.max(), high_flow_values.max())
        
        global_vmax = max(abs(global_vmin), abs(global_vmax))
        global_vmin = -global_vmax
     
        # Create a figure with two subplots (one row, two columns)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        fig.patch.set_facecolor(facecolor)
        
        # Plot low flow trends
        ax = axs[0]
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = catchments_chara['degimpact'] != 's'
        # Drop fellsá and eyjabakkafoss (15, 48)
        im = merged_gdf.loc[mask].drop([15,48], errors='ignore').plot(column='low_flow_trend', legend=False, vmin=global_vmin, vmax=global_vmax,
                                       legend_kwds={'label': 'Trend in 10th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                                    'orientation': "horizontal",
                                                    'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap, s=streamflow_markersize, zorder=2)
        significant_points = merged_gdf.loc[mask].drop([15,48], errors='ignore')[merged_gdf.loc[mask]['low_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x - shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markeredgewidth =2,
                markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5', zorder=1)
        ax.set_title('Annual low flow (10th percentile)', size=20, y=0.95)
        catchments.loc[merged_gdf['low_flow_trend'].loc[mask].drop([15,48], errors='ignore').dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        
        # Plot high flow trends
        ax = axs[1]
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        mask = catchments_chara['degimpact'] != 's'
        im = merged_gdf.loc[mask].plot(column='high_flow_trend', legend=False, vmin=global_vmin, vmax=global_vmax,
                                       legend_kwds={'label': 'Trend in 90th percentile of daily streamflow from %s-%s [%%/decade]' % (start_year, end_year),
                                                    'orientation': "horizontal",
                                                    'shrink': 0.8, 'pad': 0.03}, ax=ax, cmap=colormap, s=streamflow_markersize, zorder=2)
        significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['high_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x - shift_value, significant_points.geometry.y, marker='o', markersize=streamflow_sign_size, markeredgewidth =2,
                markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5', zorder=1)
        ax.set_title('Annual high flow (90th percentile)', size=map_fontsize, y=0.95)
        catchments.loc[merged_gdf['high_flow_trend'].loc[mask].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
    
        # Add a common colorbar below the subplots
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=global_vmin, vmax=global_vmax))
        sm.set_array([])  # Dummy array
        cax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  
        cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=global_vmin, vmax=global_vmax), orientation='horizontal')
        cb.set_label('Trend in high/low flow from %s-%s [%%/decade]' % (start_year, end_year), size=map_fontsize)
        cb.ax.tick_params(labelsize=map_fontsize)
        
        # Save the figure
        save_path = os.path.join(maps_path, 'highlow_flow_trends.png')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        
    ###################################### Plot trendlines for annual data - Modified MK and Thiel Sen ###############################################
    if which_plots['annual_series']:
        for gauge in results.index:
            #print(gauge)
            plt.figure()  # Create a new figure for each column
            plt.suptitle('Gauge %s, %s %s' % (gauge,gauges.loc[gauge]['river'], gauges.loc[gauge]['name']))
    
            # Select the valid data for the current gauge
            valid_data = valid_data_dict.get((str(gauge), 'annual'))
            try:
                trend = plot_dict_mod.get((str(gauge), 'annual'))[0]
                intercept = plot_dict_mod.get((str(gauge), 'annual'))[1]
                x_values = plot_dict_mod.get((str(gauge), 'annual'))[2]
            except:
                pass
    
            # Create a scatter plot for the selected valid data
            try:
                plt.scatter(valid_data.index, valid_data.values, label=None) 
                trend_years = pds.date_range(valid_data.index[0],valid_data.index[-1],freq='A') 
                if results.loc[gauge]['pval_mod'] < 0.05:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='-',c='r', label='%s %% per decade, pval = %s' % (np.round(results.loc[gauge]['annual_trend_ts'],1),results.loc[gauge]['pval_mod']))
                else:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='--',c='r', label='%s %% per decade, pval=%s' % (np.round(results.loc[gauge]['annual_trend_ts'],1),results.loc[gauge]['pval_mod']))
    
            except:
                invalid_data = invalid_data_dict.get((str(gauge), 'annual'))
                plt.scatter(invalid_data.index, invalid_data.values,color='lightblue', label='Trend not calculated due to missing data')#, marker='o')
    
            plt.xlabel('Year')
            plt.ylabel('Yearly avg. flow (m3/s)')
            plt.legend()
    
            # Adjust subplot layout
            plt.tight_layout()
            save_name = '%s.png' % str(gauge)
            save_path = os.path.join(annual_trends_path_mod_ts,save_name)
            plt.savefig(save_path)
            plt.close()

    ################################## Plot trendlines for seasonal means - Modified MK and Thiel Sen ##############################################
    if which_plots['seasonal_series']:
        for gauge in results.index:
            # Create a subplot with 4 plots in a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            plt.suptitle('Gauge %s, %s %s' % (gauge,gauges.loc[gauge]['river'], gauges.loc[gauge]['name']))
    
            for i, season in enumerate(seasons):
                row = i // 2
                col = i % 2
    
                # Select the valid data for the current season and gauge
                valid_data = valid_data_dict.get((str(gauge), season))
                try:
                    trend = plot_dict_mod.get((str(gauge), season))[0]
                    intercept = plot_dict_mod.get((str(gauge), season))[1]
                    x_values = plot_dict_mod.get((str(gauge), season))[2]
                except:
                    pass
    
                # Create a scatter plot for the selected valid data
                try:
                    axes[row, col].scatter(valid_data.index, valid_data.values, label=None) 
                    trend_years = pds.date_range(valid_data.index[0],valid_data.index[-1],freq='A') 
                    if results.loc[gauge][f'pval_{season}_mod'] < 0.05:
                        axes[row, col].plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='-',c='r', label='%s %% per decade, pval=%s' % (np.round(results.loc[gauge][f'trend_{season}_ts'],1),results.loc[gauge][f'pval_{season}_mod']))
                    else:
                        axes[row, col].plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='--',c='r', label='%s %% per decade, pval=%s' % (np.round(results.loc[gauge][f'trend_{season}_ts'],1),results.loc[gauge][f'pval_{season}_mod']))
                    axes[row, col].set_title(season)
                    axes[row, col].set_xlabel('Date')
                    axes[row, col].set_ylabel('Daily avg. flow (m3/s)')
                    axes[row, col].legend()
                except:
                    invalid_data = invalid_data_dict.get((str(gauge), season))
                    axes[row, col].scatter(invalid_data.index, invalid_data.values, label='Trend not calculated due to missing data',color='lightblue')#, marker='o')
                    axes[row, col].set_title(season)
                    axes[row, col].set_xlabel('Date')
                    axes[row, col].set_ylabel('Daily avg. flow (m3/s)')
                    axes[row, col].legend()
    
            # Adjust subplot layout
            plt.tight_layout()
            save_name = '%s.png' % str(gauge)
            save_path = os.path.join(seasonal_trends_path_mod_ts,save_name)
            plt.savefig(save_path)
            plt.close()

    ################################## Plot trendlines for monthly series - Modified MK and Thiel Sen ##############################################
    if which_plots['monthly_series']:
        month_names = ['Janúar','Febrúar','Mars','Apríl','Maí','Júní','Júlí','Ágúst','September','Október','Nóvember','Desember']
        months = np.arange(1,13,1)
        for gauge in results.index:
            # Create a subplot with 12 plots in a 3x4 grid
            fig, axes = plt.subplots(4, 3, figsize=(10, 8))
            plt.suptitle('Gauge %s, %s %s' % (gauge,gauges.loc[gauge]['river'], gauges.loc[gauge]['name']))
    
            for i, month in enumerate(months):
                row = i // 3
                col = i % 3
    
                # Select the valid data for the current season and gauge
                valid_data = valid_data_dict.get((str(gauge), month))
                try:
                    trend = plot_dict_mod.get((str(gauge), month))[0]
                    intercept = plot_dict_mod.get((str(gauge), month))[1]
                    x_values = plot_dict_mod.get((str(gauge), month))[2]
                except:
                    pass
    
                # Create a scatter plot for the selected valid data
                try:
                    axes[row, col].scatter(valid_data.index, valid_data.values, label=None) 
                    trend_years = pds.date_range(valid_data.index[0],valid_data.index[-1],freq='A') 
                    if results.loc[gauge][f'pval_{month}_mod'] < 0.05:
                        axes[row, col].plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='-',c='r', label='%s %% per decade' % (np.round(results.loc[gauge][f'trend_{month}_ts'],1)))
                    else:
                        axes[row, col].plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='--',c='r')
                    axes[row, col].set_title(month_names[month-1])
                    if i in [9,10,11]:
                        axes[row, col].set_xlabel('Date')
                    if i in [0,3,6,9]:
                        axes[row, col].set_ylabel('Daily avg. flow (m3/s)')
                    axes[row, col].grid()
                    rotation_angle = 45 
                    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=rotation_angle)
                except:
                    invalid_data = invalid_data_dict.get((str(gauge), month))
                    axes[row, col].scatter(invalid_data.index, invalid_data.values,color='lightblue') 
                    axes[row, col].set_title(month_names[month-1])
                    if i in [9,10,11]:
                        axes[row, col].set_xlabel('Date')
                    if i in [0,3,6,9]:
                        axes[row, col].set_ylabel('Daily avg. flow (m3/s)')
                    axes[row, col].grid()  
                    rotation_angle = 45  
                    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=rotation_angle)
    
            # Adjust subplot layout
            plt.tight_layout()
            save_name = '%s.png' % str(gauge)
            save_path = os.path.join(monthly_trends_path_mod_ts,save_name)
            plt.savefig(save_path)
            plt.close()
           
    ###################################### Plot timeseries for 10th percentile flow ###############################################
    if which_plots['low_flow_series']:
        for gauge in results.index:
            plt.figure()  # Create a new figure for each column
            plt.suptitle('Gauge %s, %s %s' % (gauge,gauges.loc[gauge]['river'], gauges.loc[gauge]['name']))
    
            # Select the valid data for the current gauge
            valid_data = valid_data_dict.get((str(gauge), 'low_flow'))
            try:
                trend = plot_dict_mod.get((str(gauge), 'low_flow'))[0]
                intercept = plot_dict_mod.get((str(gauge), 'low_flow'))[1]
                x_values = plot_dict_mod.get((str(gauge), 'low_flow'))[2]
            except:
                pass
    
            # Create a scatter plot for the selected valid data
            try:
                trend_years = pds.date_range('%s-12-31' % valid_data.index[0],'%s-12-31' %valid_data.index[-1],freq='A') 
                plt.scatter(trend_years, valid_data.values, label=None) 
                if results.loc[gauge]['low_flow_pval'] < 0.05:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='-',c='r', label='%s %% per decade, pval = %s' % (np.round(results.loc[gauge]['low_flow_trend'],1),results.loc[gauge]['low_flow_pval']))
                else:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='--',c='r', label='%s %% per decade, pval=%s' % (np.round(results.loc[gauge]['low_flow_trend'],1),results.loc[gauge]['low_flow_pval']))
    
            except:
                invalid_data = invalid_data_dict.get((str(gauge), 'annual'))
                print('lowflows: invalid')
                plt.scatter(invalid_data.index, invalid_data.values,color='lightblue', label='Trend not calculated due to missing data')#, marker='o')
    
            plt.xlabel('Date')
            plt.ylabel('Yearly 10th percentile of daily avg. flow (m3/s)')
            plt.grid()
            plt.legend()
    
            # Adjust subplot layout
            plt.tight_layout()
            save_name = 'low_flows_%s.png' % str(gauge)
            save_path = os.path.join(annual_trends_path_mod_ts,save_name)
            plt.savefig(save_path)
            plt.close()

    ###################################### Plot timeseries for 90th percentile flow ###############################################
    if which_plots['high_flow_series']:
        for gauge in results.index:
            plt.figure()  # Create a new figure for each column
            plt.suptitle('Gauge %s, %s %s' % (gauge,gauges.loc[gauge]['river'], gauges.loc[gauge]['name']))
    
            # Select the valid data for the current gauge
            valid_data = valid_data_dict.get((str(gauge), 'high_flow'))
            try:
                trend = plot_dict_mod.get((str(gauge), 'high_flow'))[0]
                intercept = plot_dict_mod.get((str(gauge), 'high_flow'))[1]
                x_values = plot_dict_mod.get((str(gauge), 'high_flow'))[2]
            except:
                pass
    
            # Create a scatter plot for the selected valid data
            try:
                trend_years = pds.date_range('%s-12-31' % valid_data.index[0],'%s-12-31' %valid_data.index[-1],freq='A') 
                plt.scatter(trend_years, valid_data.values, label=None) 
                if results.loc[gauge]['high_flow_pval'] < 0.05:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='-',c='r', label='%s %% per decade, pval = %s' % (np.round(results.loc[gauge]['high_flow_trend'],1),results.loc[gauge]['high_flow_pval']))
                else:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept,ls='--',c='r', label='%s %% per decade, pval=%s' % (np.round(results.loc[gauge]['high_flow_trend'],1),results.loc[gauge]['high_flow_pval']))
    
            except:
                invalid_data = invalid_data_dict.get((str(gauge), 'annual'))
                print('highflows: invalid')
                plt.scatter(invalid_data.index, invalid_data.values,color='lightblue', label='Trend not calculated due to missing data')#, marker='o')
    
            plt.xlabel('Date')
            plt.ylabel('Yearly 90th percentile of daily avg. flow (m3/s)')
            plt.grid()
            plt.legend()
    
            # Adjust subplot layout
            plt.tight_layout()
            save_name = 'high_flow_%s.png' % str(gauge)
            save_path = os.path.join(annual_trends_path_mod_ts,save_name)
            plt.savefig(save_path)
            plt.close()       
      