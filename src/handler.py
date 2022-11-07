import pickle 
import pandas as pd
from flask import Flask, request, Response
import insurance.insurance as insurance
model = pickle.load(open('models/insurance_cross-sell_rf_model.pkl', 'rb') ) 

#initializing API
app = Flask(__name__)

@app.route('/predict', methods=['POST'] )
def health_insurance_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): #unique example
            test_raw = pd.DataFrame(test_json, index=[0] )
        else: #multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys() )
    
        #Instantiate Insurance class
        pipeline = insurance.Insurance()

        #data cleaning
        df = pipeline.data_cleaning(test_raw)

        #creation of features
        df = pipeline.feature_engineering(df)

        #data preparation
        df = pipeline.data_preparation(df)

        #data selection
        df = pipeline.feature_selection(df)

        #prediction
        df_response = pipeline.prediction(model,transformed_data = df,original_data= test_raw)

        return df_response
    
    else: 
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
        