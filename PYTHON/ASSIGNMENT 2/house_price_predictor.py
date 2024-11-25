#Assignment 1 :
#Name :Omkar Ankush Biramane
#Roll no: 391007
#Batch:A1

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HousePricePredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = None
    
    def load_and_preprocess_data(self):

        data = pd.read_csv(self.dataset_path)
        
        self.features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
        X = data[self.features]
        y = data['SalePrice']
        
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, input_features):
       
        input_features_df = pd.DataFrame([input_features], columns=self.features)
        input_scaled = self.scaler.transform(input_features_df)
        
     
        prediction = self.model.predict(input_scaled)
        return prediction[0]
