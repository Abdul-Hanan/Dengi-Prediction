
# Abdul Hanan Ashraf 18101147


```python
import pandas as pd
import numpy as np
```


```python
dftrainfeatures=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_features_train.csv");
dftrainlabels=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_labels_train.csv");
dftest=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_features_test.csv");
```


```python
dftrainfeatures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>1990</td>
      <td>18</td>
      <td>1990-04-30</td>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>...</td>
      <td>32.00</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>1990</td>
      <td>19</td>
      <td>1990-05-07</td>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>...</td>
      <td>17.94</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>1990</td>
      <td>20</td>
      <td>1990-05-14</td>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>...</td>
      <td>26.10</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>1990</td>
      <td>21</td>
      <td>1990-05-21</td>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>...</td>
      <td>13.90</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>1990</td>
      <td>22</td>
      <td>1990-05-28</td>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>...</td>
      <td>12.20</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
dftrainfeatures.isnull().any()

```




    city                                     False
    year                                     False
    weekofyear                               False
    week_start_date                          False
    ndvi_ne                                   True
    ndvi_nw                                   True
    ndvi_se                                   True
    ndvi_sw                                   True
    precipitation_amt_mm                      True
    reanalysis_air_temp_k                     True
    reanalysis_avg_temp_k                     True
    reanalysis_dew_point_temp_k               True
    reanalysis_max_air_temp_k                 True
    reanalysis_min_air_temp_k                 True
    reanalysis_precip_amt_kg_per_m2           True
    reanalysis_relative_humidity_percent      True
    reanalysis_sat_precip_amt_mm              True
    reanalysis_specific_humidity_g_per_kg     True
    reanalysis_tdtr_k                         True
    station_avg_temp_c                        True
    station_diur_temp_rng_c                   True
    station_max_temp_c                        True
    station_min_temp_c                        True
    station_precip_mm                         True
    dtype: bool




```python
dftrainfeatures.drop('city', axis = 1, inplace = True)
dftrainfeatures.drop('year', axis = 1, inplace = True)
dftrainfeatures.drop('weekofyear', axis = 1, inplace = True)
dftrainfeatures.drop('week_start_date', axis = 1, inplace = True)
```


```python
dftrainfeatures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>297.742857</td>
      <td>292.414286</td>
      <td>299.8</td>
      <td>295.9</td>
      <td>32.00</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>298.442857</td>
      <td>293.951429</td>
      <td>300.9</td>
      <td>296.4</td>
      <td>17.94</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>298.878571</td>
      <td>295.434286</td>
      <td>300.5</td>
      <td>297.3</td>
      <td>26.10</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>299.228571</td>
      <td>295.310000</td>
      <td>301.4</td>
      <td>297.0</td>
      <td>13.90</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>299.664286</td>
      <td>295.821429</td>
      <td>301.9</td>
      <td>297.5</td>
      <td>12.20</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as s
dftrainfeatures.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x149e615f080>




![png](output_7_1.png)



```python
#Sensor data ofen has noise and ourliners we are going to drop them or replace them with mean
dftrainfeatures.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1262.000000</td>
      <td>1404.000000</td>
      <td>1434.000000</td>
      <td>1434.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1413.000000</td>
      <td>1413.000000</td>
      <td>1436.000000</td>
      <td>1442.000000</td>
      <td>1434.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.142294</td>
      <td>0.130553</td>
      <td>0.203783</td>
      <td>0.202305</td>
      <td>45.760388</td>
      <td>298.701852</td>
      <td>299.225578</td>
      <td>295.246356</td>
      <td>303.427109</td>
      <td>295.719156</td>
      <td>40.151819</td>
      <td>82.161959</td>
      <td>45.760388</td>
      <td>16.746427</td>
      <td>4.903754</td>
      <td>27.185783</td>
      <td>8.059328</td>
      <td>32.452437</td>
      <td>22.102150</td>
      <td>39.326360</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.140531</td>
      <td>0.119999</td>
      <td>0.073860</td>
      <td>0.083903</td>
      <td>43.715537</td>
      <td>1.362420</td>
      <td>1.261715</td>
      <td>1.527810</td>
      <td>3.234601</td>
      <td>2.565364</td>
      <td>43.434399</td>
      <td>7.153897</td>
      <td>43.715537</td>
      <td>1.542494</td>
      <td>3.546445</td>
      <td>1.292347</td>
      <td>2.128568</td>
      <td>1.959318</td>
      <td>1.574066</td>
      <td>47.455314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.406250</td>
      <td>-0.456100</td>
      <td>-0.015533</td>
      <td>-0.063457</td>
      <td>0.000000</td>
      <td>294.635714</td>
      <td>294.892857</td>
      <td>289.642857</td>
      <td>297.800000</td>
      <td>286.900000</td>
      <td>0.000000</td>
      <td>57.787143</td>
      <td>0.000000</td>
      <td>11.715714</td>
      <td>1.357143</td>
      <td>21.400000</td>
      <td>4.528571</td>
      <td>26.700000</td>
      <td>14.700000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.044950</td>
      <td>0.049217</td>
      <td>0.155087</td>
      <td>0.144209</td>
      <td>9.800000</td>
      <td>297.658929</td>
      <td>298.257143</td>
      <td>294.118929</td>
      <td>301.000000</td>
      <td>293.900000</td>
      <td>13.055000</td>
      <td>77.177143</td>
      <td>9.800000</td>
      <td>15.557143</td>
      <td>2.328571</td>
      <td>26.300000</td>
      <td>6.514286</td>
      <td>31.100000</td>
      <td>21.100000</td>
      <td>8.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.128817</td>
      <td>0.121429</td>
      <td>0.196050</td>
      <td>0.189450</td>
      <td>38.340000</td>
      <td>298.646429</td>
      <td>299.289286</td>
      <td>295.640714</td>
      <td>302.400000</td>
      <td>296.200000</td>
      <td>27.245000</td>
      <td>80.301429</td>
      <td>38.340000</td>
      <td>17.087143</td>
      <td>2.857143</td>
      <td>27.414286</td>
      <td>7.300000</td>
      <td>32.800000</td>
      <td>22.200000</td>
      <td>23.850000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.248483</td>
      <td>0.216600</td>
      <td>0.248846</td>
      <td>0.246982</td>
      <td>70.235000</td>
      <td>299.833571</td>
      <td>300.207143</td>
      <td>296.460000</td>
      <td>305.500000</td>
      <td>297.900000</td>
      <td>52.200000</td>
      <td>86.357857</td>
      <td>70.235000</td>
      <td>17.978214</td>
      <td>7.625000</td>
      <td>28.157143</td>
      <td>9.566667</td>
      <td>33.900000</td>
      <td>23.300000</td>
      <td>53.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.508357</td>
      <td>0.454429</td>
      <td>0.538314</td>
      <td>0.546017</td>
      <td>390.600000</td>
      <td>302.200000</td>
      <td>302.928571</td>
      <td>298.450000</td>
      <td>314.000000</td>
      <td>299.900000</td>
      <td>570.500000</td>
      <td>98.610000</td>
      <td>390.600000</td>
      <td>20.461429</td>
      <td>16.028571</td>
      <td>30.800000</td>
      <td>15.800000</td>
      <td>42.200000</td>
      <td>25.600000</td>
      <td>543.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Not that important , this is repeated mantually by putting the quartiles to get IQR ranges
q1=0.144209
q3=0.246982
IQR=1.5*(q3-q1)
q11=q1-IQR
q31=q3+IQR
print(q11)
print(q31)
```

    -0.009950500000000001
    0.40114150000000004
    


```python
dftrainfeatures["ndvi_ne"] = np.where(dftrainfeatures["ndvi_ne"] >0.5537825000000001, 0.142294,dftrainfeatures['ndvi_ne'])
dftrainfeatures["ndvi_ne"] = np.where(dftrainfeatures["ndvi_ne"] <-0.26034950000000007, 0.142294,dftrainfeatures['ndvi_ne'])

dftrainfeatures["ndvi_nw"] = np.where(dftrainfeatures["ndvi_nw"] >0.4676745, 0.130553,dftrainfeatures['ndvi_nw'])
dftrainfeatures["ndvi_nw"] = np.where(dftrainfeatures["ndvi_nw"] <-0.20185749999999997,0.130553,dftrainfeatures['ndvi_nw'])

dftrainfeatures["ndvi_se"] = np.where(dftrainfeatures["ndvi_se"] >0.3894845, 0.203783,dftrainfeatures['ndvi_se'])
dftrainfeatures["ndvi_se"] = np.where(dftrainfeatures["ndvi_se"] <0.014448500000000003,0.203783,dftrainfeatures['ndvi_se'])

dftrainfeatures["ndvi_sw"] = np.where(dftrainfeatures["ndvi_sw"] >0.40114150000000004, 0.202305,dftrainfeatures['ndvi_sw'])
dftrainfeatures["ndvi_sw"] = np.where(dftrainfeatures["ndvi_sw"] <-0.009950500000000001, 0.202305,dftrainfeatures['ndvi_sw'])
```


```python
dftrainfeatures.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x149e60b1550>




![png](output_11_1.png)



```python
dftrainlabels.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>1990</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>1990</td>
      <td>19</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>1990</td>
      <td>20</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>1990</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>1990</td>
      <td>22</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
dftrainfeatures = dftrainfeatures.fillna(dftrainfeatures.mean())
X = dftrainfeatures.iloc[:,0:22]
Y = dftrainlabels[['total_cases']]
```


```python
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
# normalize the data attributes
#values scaled between 0-1
scaler = preprocessing.MinMaxScaler()
scaler.fit(X) #X values ----- Y
normalized_X = scaler.transform(X)
```


```python
X_train, X_test, y_train, y_test = train_test_split(normalized_X, Y, test_size=0.3, 
                                                    random_state=0) 
```


```python
from sklearn.model_selection import GridSearchCV
model = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
clf = GridSearchCV(model, parameters, cv=10, verbose=0, n_jobs=-1, refit=True)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)



```




    0.13094830603933916




```python
from sklearn import ensemble
params = {'n_estimators': 600, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.02, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)
gbr.score(X_train,y_train)
```

    D:\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    0.9320673478463184




```python
predictionsLR = clf.predict(X_test)
predictionsGB = gbr.predict(X_test)
```


```python
from sklearn.metrics import mean_absolute_error
```


```python
mean_absolute_error(y_test, predictionsLR)

```




    21.424089518003214




```python
mean_absolute_error(y_test, predictionsGB)
```




    20.035519132009487




```python
predictionsGB
```




    array([ 3.18685989e+01,  4.79966878e+00,  3.14628516e+02,  2.04366296e+01,
            1.33918781e+01,  1.61357604e+01,  3.38590378e+01,  1.63331033e+01,
            5.17471417e+01,  1.54774782e+01,  1.33033127e+01,  1.58217469e+01,
            3.11428302e+01,  4.15348570e+01,  7.31203479e+00,  3.74435822e+01,
            1.37700402e+01,  4.07867691e+01,  2.93732729e+00,  3.06005522e+01,
            1.75001415e+01,  2.63929243e+00,  3.32024892e+01,  2.38931593e+01,
            1.57292521e+01,  7.69036654e+00, -5.60772744e-01,  4.66403922e+01,
            5.43557032e+00,  1.71637749e+01,  8.63577602e+00,  2.47623624e+01,
            1.74962791e+01,  4.60477087e+01,  3.97063913e+01,  1.19070149e+01,
            8.36606649e+00,  4.52033380e+00,  2.28244323e+02,  3.06248973e+01,
            1.59348660e+01,  2.92627614e+01,  9.14899337e+00,  2.46482749e+01,
            8.07559413e+00,  2.22625481e+01,  3.31103610e+00,  4.82282297e+01,
            3.16795260e-01,  1.48666941e+01,  7.53382516e+00,  1.21956457e+01,
            1.02384488e+02,  2.67452901e+00,  1.56507703e+01,  3.40306670e+01,
            2.36490178e+01,  3.26651550e+01,  5.90845522e+01,  4.32019446e+00,
            1.87148947e+01,  1.34007237e+01,  1.50199716e+01,  3.11491702e+01,
            2.52564916e+01,  8.92441463e+00, -1.45848982e+00,  4.01049278e+00,
            8.23765511e+00,  3.25265142e+01,  1.99656642e+01,  6.43034021e+01,
            5.26529939e+01,  2.13310090e+01,  1.18535512e+01,  1.01422382e+01,
            1.86439822e+01,  1.73730772e+01,  6.04420676e+00,  3.83755615e+01,
            1.12155458e+01,  1.60772932e+01,  2.06397399e+01,  2.88643695e+01,
            3.12824958e+01,  3.32198589e+01,  9.42723894e+00,  1.63165241e+01,
            5.81774825e+00,  4.28756693e+00,  1.79027353e+00,  5.32318329e+01,
            1.42627193e+01,  1.19314031e+01,  5.18047059e+01,  3.77721921e+01,
            2.37480620e+01,  3.56365498e+01,  2.62839187e+01,  1.83926366e+01,
            1.54512308e+01,  2.96010192e+01,  5.14286046e+01,  7.25997843e+00,
            7.93306973e+00,  7.71336707e+00,  2.22463616e+01,  4.56480925e+01,
            3.32009881e+01,  1.35262807e+01,  7.37584844e+01,  8.29872932e+00,
            6.94747189e+00,  3.78202797e+01,  7.01536433e+00,  1.82793787e+01,
            9.24023176e+00,  3.64994394e+00,  5.76793048e+01,  5.85460252e+00,
            6.20869919e+00,  4.18461295e+00,  4.96011809e+01,  1.64115369e+01,
            1.03032557e+01,  1.73748000e+01,  8.48974479e+00,  2.43840921e+01,
            3.16787363e+00,  6.56541587e+00,  5.44573035e+01,  2.45718040e+01,
            4.23950727e+01,  9.60882485e+00,  8.15391337e+00,  5.33940649e+01,
            1.81239505e+00,  5.11631055e+01,  1.49112665e+01,  5.47153370e+00,
            5.10370194e+01,  1.23143357e+01,  6.00389524e+00,  5.48315519e+00,
            2.79392806e+01,  1.16841554e+01,  8.38177892e+00,  2.58707725e+01,
            2.04461102e+01,  8.00942767e+00,  2.48788801e+00,  3.99726357e+00,
            5.63687807e+00,  1.57820039e+01,  2.40354300e+01,  3.41646328e+01,
            3.30216418e+01,  4.87024511e+01,  2.37965515e+01,  2.41548025e+00,
            2.14540426e+01,  2.27030640e+01,  7.82546529e+00,  1.74094757e+01,
            3.08232822e+01,  1.84198539e+01,  1.27473917e+01,  2.85598415e+01,
            1.41150226e+01,  1.50972855e+01,  3.05979707e+01,  2.85209702e+01,
            4.44167328e+01,  3.30857130e+01,  1.62249397e+01,  3.05569030e+01,
            8.90954553e+01,  4.96782007e+00,  2.37001033e+01,  3.47798680e+02,
            3.21076169e+01,  8.14349275e+00,  1.04401696e+01,  3.78745193e+00,
            4.13157099e+00,  1.13252461e+01,  2.37956832e+01,  3.28601577e+01,
            3.72747422e+00,  1.38939896e+01,  2.80954190e+01,  6.35882591e+01,
            9.39347904e+00,  5.76766560e+01,  2.59829179e+01,  2.14549805e+01,
            2.19635158e+01,  2.60679190e+01,  2.52854703e+01,  1.15245390e+01,
            5.63203730e+01,  4.41879443e+01,  2.15204546e+01,  1.95374614e+02,
            1.00495894e+01,  1.65901802e+01,  5.97197636e+00,  1.79114408e+01,
            1.29619706e+02,  8.43784292e+00,  3.75559307e+01,  2.57744305e+01,
            4.16149509e+01,  2.12804438e+01,  3.28147581e+01,  2.09310368e+01,
            1.21563320e+01,  6.01901736e+00,  2.95667654e+01,  1.19981498e+01,
            8.03248250e+00,  6.61643772e+01,  1.15768308e+01,  3.08526150e+01,
            2.37418322e+01,  4.55831380e+01,  2.11115848e+01,  1.33315898e+01,
            2.88885148e+00,  3.21197409e+01,  1.96942108e+01,  9.61770548e+00,
            6.21460829e-01,  4.10914828e+00,  1.91256378e+01,  2.03474437e+01,
            2.33881106e+01,  3.15638882e+01,  2.88463069e+01,  4.78107814e+01,
            6.64829922e+00,  1.71162889e+01,  2.75710701e+02,  4.70175755e+00,
            2.14849243e+01,  2.05579585e+01,  1.76591086e+02,  6.14459318e+01,
            3.89333620e+01,  4.92183985e+01,  3.35788876e+00,  8.77097545e+00,
            1.08379182e+01,  3.19326112e+01,  2.81440644e+01,  2.60685197e+01,
            2.63505743e+01,  3.55245896e+01,  3.16495489e+01,  2.22409873e+01,
            1.28886193e-01,  1.38027265e+01,  2.54439293e+01,  2.25034899e+01,
            3.40882490e+01,  2.40515380e+01,  1.94890586e+01,  6.41565740e+00,
            6.72144737e+00,  2.93191463e+01,  1.93613924e+01,  6.13133076e+01,
            8.86500471e+00,  8.18516740e+00,  3.28727892e+01,  1.47536503e+01,
            1.11711159e+01,  5.72704437e+01,  2.66551918e+01,  1.37260124e+01,
            1.39367575e+01,  2.48482641e+01,  9.43436615e+01,  2.92659504e+01,
            3.38805239e+01,  1.65085164e+01,  8.94602309e+00,  2.55609777e+01,
            1.03126417e+01,  1.41844570e+01,  1.20628963e+01,  2.79492943e+01,
            1.06399707e+01,  1.18126553e+01,  2.53584392e+01,  2.71286830e+01,
            1.93648399e+01,  4.76905511e+00,  2.29664289e+01,  2.68153876e+01,
            7.95690114e+01,  5.10111639e+01,  1.79717084e+01,  6.19492325e+01,
            2.43361428e+01,  1.39511386e+01,  4.24837029e+01,  8.43548393e+00,
            2.93916373e+01,  3.28278663e+01,  3.09366682e+01,  2.26600599e+01,
            1.94128530e+01,  1.73333447e+01,  1.02862064e+01,  1.51527357e+01,
            2.79823173e+01, -1.64206371e+00,  1.83672199e+01,  1.69576440e+01,
            5.02239085e+01,  1.41548136e+01,  2.82096807e+01,  3.09907763e+00,
            1.38544505e+01,  1.82916778e+01,  2.10110875e+01,  1.78538755e+01,
            1.94768428e+01,  8.34452525e+00,  4.00565866e+01,  1.98139613e+01,
            2.09972046e+01,  8.85812255e+01,  1.02612716e+01,  2.10031952e+01,
            7.04334832e+01,  4.32946358e+00,  6.39560032e+00,  4.58943002e+01,
           -4.54763379e+00,  2.15152273e+01,  6.33592964e+00,  2.09710426e+01,
            1.54785883e+01,  6.84530744e+00,  1.42517736e+01,  1.84674138e+00,
            5.07089031e+01,  2.30723097e+01,  1.25381330e+01,  2.42601703e+01,
            2.56058108e+00,  2.43467869e+01,  6.84639506e+00,  1.92041826e+01,
            7.14454476e+00,  8.62254492e+00,  8.85072296e+00,  5.21278129e+01,
            4.19323011e+00,  9.11991220e+00,  8.92490669e+00,  1.61896172e+01,
            2.50272432e+01,  2.37275729e+01,  3.60046576e+00,  1.85679595e+01,
            1.35043544e+01,  2.89698960e+01,  7.47432881e+01,  4.42100906e+00,
            1.33735362e+01,  5.82215415e+00,  1.51093861e+01,  1.89815350e+01,
            4.31149581e+00,  3.56275200e+00,  6.87566312e+00,  3.99744529e+01,
            1.13979578e+01,  1.29295684e+01,  2.42880207e+01,  1.90096457e+01,
            2.43781020e+01,  2.88612706e+01,  6.40566003e+00,  8.02156507e+00,
            1.09693106e+01,  3.56718003e+01,  1.10642577e+01,  3.52344305e+00,
            7.03662180e+01,  3.28488990e+00,  2.39505491e+01,  8.55449531e+00,
            2.15098492e+01,  6.58674706e+00,  3.55151604e+01,  4.01138452e+01,
            8.67651379e+00,  4.24226573e+01,  2.09371898e+01,  1.31054826e+00,
            5.42363524e+01,  3.07334702e+01,  3.00187832e+01,  2.90778899e+01,
            3.50511480e+01,  1.50295851e+01,  3.31287076e+01,  2.07762000e+01,
            2.35220131e+01,  1.35614443e+01,  4.53860838e+00,  3.09299308e+01,
            9.35859176e+00,  1.55433418e+01,  3.41298566e+01,  9.86100921e+00,
            4.75860328e+00,  8.72641826e+00,  6.18707546e+01,  5.94677396e+01,
            1.01358479e+02,  8.43164487e+00,  8.18642138e+00,  3.01957023e+01,
            1.54662253e+01,  1.08108119e+01,  2.02547851e+01,  1.27235387e+01,
            3.52981551e+01,  2.48960390e+01,  1.89009730e+01,  2.55946480e+01,
            2.49524128e+01])




```python
dftest.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.00000</td>
      <td>416.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.127630</td>
      <td>0.125514</td>
      <td>0.205477</td>
      <td>0.198421</td>
      <td>38.354324</td>
      <td>298.818295</td>
      <td>299.353071</td>
      <td>295.419179</td>
      <td>303.623430</td>
      <td>295.743478</td>
      <td>42.171135</td>
      <td>82.499810</td>
      <td>38.354324</td>
      <td>16.927088</td>
      <td>5.124569</td>
      <td>27.369587</td>
      <td>7.810991</td>
      <td>32.534625</td>
      <td>22.36855</td>
      <td>34.278589</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.152884</td>
      <td>0.137152</td>
      <td>0.075647</td>
      <td>0.086740</td>
      <td>35.086274</td>
      <td>1.465956</td>
      <td>1.303082</td>
      <td>1.519424</td>
      <td>3.094333</td>
      <td>2.754448</td>
      <td>48.791517</td>
      <td>7.360442</td>
      <td>35.086274</td>
      <td>1.554110</td>
      <td>3.534323</td>
      <td>1.214656</td>
      <td>2.414041</td>
      <td>1.913475</td>
      <td>1.71256</td>
      <td>34.446562</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.364000</td>
      <td>-0.211800</td>
      <td>0.006200</td>
      <td>-0.014671</td>
      <td>0.000000</td>
      <td>294.554286</td>
      <td>295.235714</td>
      <td>290.818571</td>
      <td>298.200000</td>
      <td>286.200000</td>
      <td>0.000000</td>
      <td>64.920000</td>
      <td>0.000000</td>
      <td>12.537143</td>
      <td>1.485714</td>
      <td>24.157143</td>
      <td>4.042857</td>
      <td>27.200000</td>
      <td>14.20000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.008762</td>
      <td>0.017188</td>
      <td>0.148677</td>
      <td>0.134132</td>
      <td>8.225000</td>
      <td>297.751429</td>
      <td>298.326786</td>
      <td>294.350000</td>
      <td>301.475000</td>
      <td>293.500000</td>
      <td>9.490000</td>
      <td>77.400000</td>
      <td>8.225000</td>
      <td>15.795714</td>
      <td>2.453571</td>
      <td>26.528571</td>
      <td>5.942857</td>
      <td>31.100000</td>
      <td>21.20000</td>
      <td>9.175000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.127630</td>
      <td>0.099163</td>
      <td>0.204479</td>
      <td>0.186486</td>
      <td>31.495000</td>
      <td>298.564286</td>
      <td>299.328571</td>
      <td>295.817857</td>
      <td>302.800000</td>
      <td>296.250000</td>
      <td>25.900000</td>
      <td>80.352857</td>
      <td>31.495000</td>
      <td>17.321429</td>
      <td>2.914286</td>
      <td>27.433333</td>
      <td>6.728571</td>
      <td>32.800000</td>
      <td>22.20000</td>
      <td>23.950000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.248361</td>
      <td>0.241050</td>
      <td>0.253364</td>
      <td>0.249875</td>
      <td>57.717500</td>
      <td>300.238214</td>
      <td>300.521429</td>
      <td>296.642143</td>
      <td>305.800000</td>
      <td>298.225000</td>
      <td>56.225000</td>
      <td>87.978214</td>
      <td>57.717500</td>
      <td>18.172500</td>
      <td>8.171429</td>
      <td>28.278571</td>
      <td>9.750000</td>
      <td>33.900000</td>
      <td>23.30000</td>
      <td>47.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.500400</td>
      <td>0.464800</td>
      <td>0.407414</td>
      <td>0.430414</td>
      <td>169.340000</td>
      <td>301.935714</td>
      <td>303.328571</td>
      <td>297.794286</td>
      <td>314.100000</td>
      <td>299.700000</td>
      <td>301.400000</td>
      <td>97.982857</td>
      <td>169.340000</td>
      <td>19.598571</td>
      <td>14.485714</td>
      <td>30.271429</td>
      <td>14.725000</td>
      <td>38.400000</td>
      <td>26.70000</td>
      <td>212.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dftest.drop('city', axis = 1, inplace = True)
dftest.drop('year', axis = 1, inplace = True)
dftest.drop('weekofyear', axis = 1, inplace = True)
dftest.drop('week_start_date', axis = 1, inplace = True)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-227-5c580f3626a9> in <module>()
    ----> 1 dftest.drop('city', axis = 1, inplace = True)
          2 dftest.drop('year', axis = 1, inplace = True)
          3 dftest.drop('weekofyear', axis = 1, inplace = True)
          4 dftest.drop('week_start_date', axis = 1, inplace = True)
    

    D:\Anaconda3\lib\site-packages\pandas\core\frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3695                                            index=index, columns=columns,
       3696                                            level=level, inplace=inplace,
    -> 3697                                            errors=errors)
       3698 
       3699     @rewrite_axis_style_signature('mapper', [('copy', True),
    

    D:\Anaconda3\lib\site-packages\pandas\core\generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3109         for axis, labels in axes.items():
       3110             if labels is not None:
    -> 3111                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       3112 
       3113         if inplace:
    

    D:\Anaconda3\lib\site-packages\pandas\core\generic.py in _drop_axis(self, labels, axis, level, errors)
       3141                 new_axis = axis.drop(labels, level=level, errors=errors)
       3142             else:
    -> 3143                 new_axis = axis.drop(labels, errors=errors)
       3144             result = self.reindex(**{axis_name: new_axis})
       3145 
    

    D:\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in drop(self, labels, errors)
       4402             if errors != 'ignore':
       4403                 raise KeyError(
    -> 4404                     '{} not found in axis'.format(labels[mask]))
       4405             indexer = indexer[~mask]
       4406         return self.delete(indexer)
    

    KeyError: "['city'] not found in axis"



```python
# Not that important , this is repeated mantually by putting the quartiles to get IQR ranges
q1=0.134079
q3=0.253243

IQR=1.5*(q3-q1)
q11=q1-IQR
q31=q3+IQR
print(q11)
print(q31)
```

    -0.044666999999999984
    0.43198899999999996
    


```python
dftest["ndvi_ne"] = np.where(dftest["ndvi_ne"] >0.6605725, 0.126050,dftest['ndvi_ne'])
dftest["ndvi_ne"] = np.where(dftest["ndvi_ne"] <-0.3987435, 0.126050,dftest['ndvi_ne'])

dftest["ndvi_nw"] = np.where(dftest["ndvi_nw"] >0.5820375, 0.126803,dftest['ndvi_nw'])
dftest["ndvi_nw"] = np.where(dftest["ndvi_nw"] <-0.3236625, 0.126803,dftest['ndvi_nw'])

dftest["ndvi_se"] = np.where(dftest["ndvi_se"] >0.41417250000000005, 0.207702,dftest['ndvi_se'])
dftest["ndvi_se"] = np.where(dftest["ndvi_se"] <-0.010631500000000044,0.207702,dftest['ndvi_se'])

dftest["ndvi_sw"] = np.where(dftest["ndvi_sw"] >0.43198899999999996, 0.201721,dftest['ndvi_sw'])
dftest["ndvi_sw"] = np.where(dftest["ndvi_sw"] <-0.044666999999999984,0.201721,dftest['ndvi_sw'])
```


```python
dftest.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x149e62787b8>




![png](output_27_1.png)



```python
dftest = dftest.fillna(dftest.mean())
```


```python
scaler = preprocessing.MinMaxScaler()
scaler.fit(dftest) #X values ----- Y
normalized_test = scaler.transform(dftest)
```


```python
dftest.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416.00000</td>
      <td>416.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.127630</td>
      <td>0.125514</td>
      <td>0.205477</td>
      <td>0.198421</td>
      <td>38.354324</td>
      <td>298.818295</td>
      <td>299.353071</td>
      <td>295.419179</td>
      <td>303.623430</td>
      <td>295.743478</td>
      <td>42.171135</td>
      <td>82.499810</td>
      <td>38.354324</td>
      <td>16.927088</td>
      <td>5.124569</td>
      <td>27.369587</td>
      <td>7.810991</td>
      <td>32.534625</td>
      <td>22.36855</td>
      <td>34.278589</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.152884</td>
      <td>0.137152</td>
      <td>0.075647</td>
      <td>0.086740</td>
      <td>35.086274</td>
      <td>1.465956</td>
      <td>1.303082</td>
      <td>1.519424</td>
      <td>3.094333</td>
      <td>2.754448</td>
      <td>48.791517</td>
      <td>7.360442</td>
      <td>35.086274</td>
      <td>1.554110</td>
      <td>3.534323</td>
      <td>1.214656</td>
      <td>2.414041</td>
      <td>1.913475</td>
      <td>1.71256</td>
      <td>34.446562</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.364000</td>
      <td>-0.211800</td>
      <td>0.006200</td>
      <td>-0.014671</td>
      <td>0.000000</td>
      <td>294.554286</td>
      <td>295.235714</td>
      <td>290.818571</td>
      <td>298.200000</td>
      <td>286.200000</td>
      <td>0.000000</td>
      <td>64.920000</td>
      <td>0.000000</td>
      <td>12.537143</td>
      <td>1.485714</td>
      <td>24.157143</td>
      <td>4.042857</td>
      <td>27.200000</td>
      <td>14.20000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.008762</td>
      <td>0.017188</td>
      <td>0.148677</td>
      <td>0.134132</td>
      <td>8.225000</td>
      <td>297.751429</td>
      <td>298.326786</td>
      <td>294.350000</td>
      <td>301.475000</td>
      <td>293.500000</td>
      <td>9.490000</td>
      <td>77.400000</td>
      <td>8.225000</td>
      <td>15.795714</td>
      <td>2.453571</td>
      <td>26.528571</td>
      <td>5.942857</td>
      <td>31.100000</td>
      <td>21.20000</td>
      <td>9.175000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.127630</td>
      <td>0.099163</td>
      <td>0.204479</td>
      <td>0.186486</td>
      <td>31.495000</td>
      <td>298.564286</td>
      <td>299.328571</td>
      <td>295.817857</td>
      <td>302.800000</td>
      <td>296.250000</td>
      <td>25.900000</td>
      <td>80.352857</td>
      <td>31.495000</td>
      <td>17.321429</td>
      <td>2.914286</td>
      <td>27.433333</td>
      <td>6.728571</td>
      <td>32.800000</td>
      <td>22.20000</td>
      <td>23.950000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.248361</td>
      <td>0.241050</td>
      <td>0.253364</td>
      <td>0.249875</td>
      <td>57.717500</td>
      <td>300.238214</td>
      <td>300.521429</td>
      <td>296.642143</td>
      <td>305.800000</td>
      <td>298.225000</td>
      <td>56.225000</td>
      <td>87.978214</td>
      <td>57.717500</td>
      <td>18.172500</td>
      <td>8.171429</td>
      <td>28.278571</td>
      <td>9.750000</td>
      <td>33.900000</td>
      <td>23.30000</td>
      <td>47.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.500400</td>
      <td>0.464800</td>
      <td>0.407414</td>
      <td>0.430414</td>
      <td>169.340000</td>
      <td>301.935714</td>
      <td>303.328571</td>
      <td>297.794286</td>
      <td>314.100000</td>
      <td>299.700000</td>
      <td>301.400000</td>
      <td>97.982857</td>
      <td>169.340000</td>
      <td>19.598571</td>
      <td>14.485714</td>
      <td>30.271429</td>
      <td>14.725000</td>
      <td>38.400000</td>
      <td>26.70000</td>
      <td>212.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = gbr.predict(normalized_test)
rounded = [np.round(x) for x in predictions]
```


```python
pd.DataFrame(rounded, columns=['predictions']).to_csv('prediction.csv')

```


```python

```


```python

```


```python

```
