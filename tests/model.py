#!/usr/bin/env python
# coding: utf-8
import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def feature_engineering(data):
    # Encoding
    le1 = LabelEncoder()
    data['category'] = le1.fit_transform(data['category'])

    le2 = LabelEncoder()
    data['channel'] = le2.fit_transform(data['channel'])

    # Scaling
    sc1 = MinMaxScaler()
    data['price'] = sc1.fit_transform(data[['price']])

    sc2 = MinMaxScaler()
    data['discount_perc'] = sc2.fit_transform(data[['discount_perc']])
    
    return data, le1, le2, sc1, sc2


def train(data, save_path):
    
    # Read data
    data = pd.DataFrame(data)

    # Feature engineering
    data, le1, le2, sc1, sc2 = feature_engineering(data)

    # Split train test
    X = data[['category', 'price', 'promotion', 'discount_perc', 'channel']]
    y = data['purchased']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict test set
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Save model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(model, open(save_path+'/model.sav', 'wb'))
    pickle.dump(le1, open(save_path+'/encoder1.sav', 'wb'))
    pickle.dump(le2, open(save_path+'/encoder2.sav', 'wb'))
    pickle.dump(sc1, open(save_path+'/scaler1.sav', 'wb'))
    pickle.dump(sc2, open(save_path+'/scaler2.sav', 'wb'))
    print(acc)

    return acc


def predict(data, model_path):
    # Load model
    model = pickle.load(open(model_path+'/model.sav', 'rb'))
    le1 = pickle.load(open(model_path+'/encoder1.sav', 'rb'))
    le2 = pickle.load(open(model_path+'/encoder2.sav', 'rb'))
    sc1 = pickle.load(open(model_path+'/scaler1.sav', 'rb'))
    sc2 = pickle.load(open(model_path+'/scaler2.sav', 'rb'))

    # Applied feature engineering
    data = [[le1.transform(np.array(data['category']).ravel())[0],
            sc1.transform(np.array(data['price']).reshape(-1,1))[0][0],
            data['promotion'],
            sc2.transform(np.array(data['discount_perc']).reshape(-1,1))[0][0],
            le2.transform(np.array(data['channel']).ravel())[0]]]

    # Predict
    result = model.predict(data)

    if result == 0:
        result = 'Failed'
    elif result == 1:
        result = 'Success'
        
    return result




