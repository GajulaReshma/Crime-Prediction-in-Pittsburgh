import numpy as np
import pandas as pd
from sklearn.externals import joblib
# import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

try:
    from tqdm import tqdm
except:
    os.system('pip install tqdm')
import time

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# from joblib import Parallel, delayed
import multiprocessing

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

arrest_data = pd.read_csv('edited_blotter_data.csv')
all_zip =  np.load('all_zip.npy')
zip_list = np.load('relevant_zip.npy')
arrest_data.drop(arrest_data.columns[0],axis = 1,inplace = True)

arrest_data.INCIDENTTIME = pd.to_datetime(arrest_data.INCIDENTTIME)
arrest_data['ZIP'] = all_zip
arrest_data = arrest_data[arrest_data.ZIP.isin(zip_list)]
arrest_data['year_month'] = pd.to_datetime(arrest_data['INCIDENTTIME']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
arrest_data_month_grouped = arrest_data.groupby(['ZIP','year_month']).agg({'PK':'count'})
arrest_data_month_grouped = arrest_data_month_grouped.add_suffix('_Count').reset_index()
arrest_data_month_grouped['month'] = (arrest_data_month_grouped.year_month).apply(
    lambda x : pd.to_datetime(x[:4]+"_"+x[5:],format = '%Y_%m'))

## Neighborhood 1: Allegheny center

def get_predictions(arrest_data_month_grouped,zip_,steps,p=12,d=1,q=1,visualize=False):

    grouped_neighborhood = arrest_data_month_grouped[arrest_data_month_grouped.ZIP == zip_].sort_values('month')

    model = None
    try:
        model = ARIMA(grouped_neighborhood.PK_Count.values,order=(p,d,q))
        fit = model.fit(disp=0)
    except:
        q = 0
        model = ARIMA(grouped_neighborhood.PK_Count.values,order=(p,d,q))
        fit = model.fit(disp=0)

    forecast = fit.forecast(steps)

    if visualize:
        plt.figure(figsize=(15,7))
        plt.plot(range(len(grouped_neighborhood)),grouped_neighborhood.PK_Count.values)
        plt.plot(range(len(grouped_neighborhood)-1,len(grouped_neighborhood)+len(forecast[0])-1),forecast[0])
        plt.title('Prediction of Total Crimes for zipcode {}'.format(str(zip_)))
        plt.fill_between(range(len(grouped_neighborhood)-1,len(grouped_neighborhood)+len(forecast[0])-1),forecast[2][:,0],forecast[2][:,1],alpha=0.4)
        plt.axvline(x = len(grouped_neighborhood) - 1,color='r')
        plt.xlabel('Number of Months')
        plt.ylabel('Number of Crimes')
        plt.grid()
        plt.savefig('images/predictions_'+str(zip_)+'.png',bbox_inches='tight')
        plt.show()

        # Lags of the original data
        plot_acf(grouped_neighborhood.PK_Count,lags=36)
#         plt.savefig('images/acf_original_'+str(zip_)+'.png',bbox_inches='tight')
        plot_pacf(grouped_neighborhood.PK_Count,lags=36)
        # Lags of the differenced data
        plot_acf(grouped_neighborhood.PK_Count.diff().dropna(),lags=36)
#         plt.savefig('images/acf_stationary_'+str(zip_)+'.png',bbox_inches='tight')
        plot_pacf(grouped_neighborhood.PK_Count.diff().dropna(),lags=36)

    return forecast

# forecast_track = []
# zip_track = []
# for zip_ in tqdm(arrest_data_month_grouped.ZIP.unique()):
#     zip_track.append(zip_)
#     try:
#         forecast_track.append(float(get_predictions(arrest_data_month_grouped,zip_,1)[0]))
#     except:
#         try:
#             forecast_track.append(float(arrest_data_month_grouped,get_predictions(zip_,1,q=0)[0]))
#         except:
#             forecast_track.append(-1)
#
# print ('Time taken:',time.time()-start,'s.')

# forecast_df = pd.DataFrame()
# forecast_df['Rank'] = range(1,len(zip_track)+1)
# forecast_df['Zip_Code'] = zip_track
# forecast_df['Incidents'] = list(map(int,forecast_track))
#
# forecast_df = forecast_df.sort_values('Incidents')[::-1]
# forecast_df['Rank'] = range(1,len(zip_track)+1)
# forecast_df

train_data = arrest_data_month_grouped[arrest_data_month_grouped.month.dt.year < 2014]
validation_data = arrest_data_month_grouped[arrest_data_month_grouped.month.dt.year == 2014]
test_data = arrest_data_month_grouped[arrest_data_month_grouped.month.dt.year == 2015]

def one_fold_cv(train_data, validation_data, test_data, zip_):
    best_mse = 1e6
    mse = []
    r2 = []
    for p in tqdm(range(11,19)):
        predictions = get_predictions(train_data,zip_,12,p=p)[0]
        actual = validation_data[validation_data.ZIP == zip_].PK_Count.values
        mse_value = MSE(predictions,actual)
        r2_value = R2(predictions,actual)
        mse.append(mse_value)
        r2.append(r2_value)

        if mse_value < best_mse:
            next_prediction = predictions[0]
            best_mse = mse_value
            best_r2 = r2_value
            best_p = p

    test_prediction = get_predictions(pd.concat([train_data,validation_data]),zip_,12,p=best_p)
    test_mse = MSE(test_data[test_data.ZIP == zip_].PK_Count.values,test_prediction[0])
    test_r2 =  R2(test_data[test_data.ZIP == zip_].PK_Count.values,test_prediction[0])

    return zip_,best_p,test_prediction,mse,r2,test_mse,test_r2

def train_models(train_data,validation_data,parallel = True):
    predictions_df = pd.DataFrame()
    results = []

    if parallel:
        os.system('pip install joblib')
        from joblib import Parallel, delayed
        import multiprocessing
        results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(one_fold_cv)(train_data, validation_data, test_data, zip_) for zip_ in tqdm(zip_list))
    else:

        for zip_ in tqdm(zip_list):
            ## This loop takes time
            results.append(one_fold_cv(train_data,validation_data,test_data,zip_))

    return results

start = time.time()
predictions = train_models(train_data,validation_data, parallel = True)
print ('Total time taken:',time.time()-start)

joblib.dump(predictions,'results.pkl')
