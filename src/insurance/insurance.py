import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, metrics
from sklearn.metrics import make_scorer
import math
import pickle
import inflection


class Insurance(object):
    def __init__(self):
        self.home_path = ''
        self.vintage_scaler = pickle.load(open(self.home_path + 'features/vintage_scaler.pkl', 'rb') )
        self.annual_premium_scaler = pickle.load(open(self.home_path + 'features/annual_premium_scaler.pkl', 'rb' ) )
        self.annual_premium_per_month_scaler = pickle.load(open(self.home_path + 'features/annual_premium_per_month_scaler.pkl', 'rb') )
        self.annual_premium_and_vintage_scaler = pickle.load(open(self.home_path + 'features/annual_premium_and_vintage_scaler.pkl', 'rb') )
        self.age_scaler = pickle.load(open(self.home_path + 'features/age.pkl', 'rb') )
        self.one_hot_encoding_scaler = pickle.load(open(self.home_path + 'features/one_hot_encoding_scaler.pkl', 'rb') )
        
    def data_cleaning(self,df):
        cols_old = df.columns.values

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))

        df.columns=cols_new

        df['driving_license'] = df['driving_license'].apply(lambda x: 'Yes' if x == 1 else 'No')
        df['previously_insured'] = df['previously_insured'].apply(lambda x: 'Yes' if x == 1 else 'No')

        return df

    
    def feature_engineering(self,df):
        df['annual_premium_and_vintage'] = df['annual_premium'] / df['vintage']
        
        return df
        
    
        
    def data_preparation(self,df):
        le = LabelEncoder()
        
        #Region code and policy_sales_channel
        df['region_code'] = le.fit_transform(df['region_code'])
        df['policy_sales_channel'] = le.fit_transform(df['policy_sales_channel'])
        
        #Gender, previously_insured,vehicle_damage,driver_license - one hot encoding
        transformed = self.one_hot_encoding_scaler.transform(df)
        transformed_df = pd.DataFrame(transformed, columns=self.one_hot_encoding_scaler.get_feature_names_out(), index=df.index)
        df = pd.concat([df,transformed_df],axis=1).drop(['gender', 'previously_insured','vehicle_damage','driving_license'], axis=1)

        #Rescaling - MinMaxScaler
        df['vintage'] = self.vintage_scaler.transform(df[['vintage']].values)
        df['annual_premium_and_vintage'] = self.annual_premium_and_vintage_scaler.transform(df[['annual_premium_and_vintage']].values)
        df['age'] = self.age_scaler.transform(df[['age']].values)

        #Rescaling - RobustScaler
        df['annual_premium'] = self.annual_premium_scaler.transform(df[['annual_premium']].values)

        #Age,vehicle_age and annual_premium - ordinal encoding
        vehicle_age_categories = {'< 1 Year':1,'1-2 Year':2, '> 2 Years':3}
        df['vehicle_age'] = df['vehicle_age'].map(vehicle_age_categories)
        
        return df
            
    def feature_selection(self,df):
        cols_selected = ['age','vehicle_age','annual_premium_and_vintage',
         'onehotencoder__vehicle_damage_Yes','onehotencoder__vehicle_damage_No',
         'policy_sales_channel','region_code', 'annual_premium','vintage']
        df = df[cols_selected]
        
        return df
    
    def prediction(self, model,transformed_data,original_data):
        pred = model.predict_proba(transformed_data)
        original_data['score'] = pred[:,1]
        original_data = original_data.sort_values(by='score', ascending=False)
        original_data['score'] = original_data['score'].apply(lambda x: str(round(x*100,2) ) + '%' )
            
        return original_data.to_json(orient='records', date_format='iso')