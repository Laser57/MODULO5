from asyncore import write
from cmath import sqrt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def transformarSerieADataset(serie, elementosPorMuestra):
    dataset = None
    salidasDataset = None
    for counter in range (len(serie)-elementosPorMuestra-1):        
        muestra = np.array([serie[counter:counter+elementosPorMuestra]])        
        salida = np.array([serie[counter+elementosPorMuestra]])
        if dataset is None:
            dataset = muestra
        else:
            dataset = np.append(dataset,muestra,axis = 0)
        if salidasDataset is None:
            salidasDataset = salida    
        else:        
            salidasDataset = np.append(salidasDataset,salida)
    return dataset, salidasDataset

def MLpolynomial(Data,modelo,dimension=2):
    adj_close = Data
    #X, Y = transformarSerieADataset(adj_close, elementosPorMuestra = 10)
    #Preparamos la informacion para procesar solo necesitamos precio ajustado y columna forecast
    lag=4
    adj_close[['forecast']] = adj_close[['Adj Close']].shift(-lag)
    adj_close=adj_close.drop(['Open'], axis = 1)
    adj_close=adj_close.drop(['High'], axis = 1)
    adj_close=adj_close.drop(['Low'], axis = 1)
    adj_close=adj_close.drop(['Close'], axis = 1)
    adj_close=adj_close.drop(['Volume'], axis = 1)
    adj_close=adj_close.drop(['Tiker'], axis = 1)

    X=np.array(adj_close)
    X=X[:adj_close.shape[0] - lag]
    #print("X INICIAL: {}".format(X))
    Y = np.array(adj_close[['forecast']])
    Y = Y[:-lag].ravel()
    #Preparamos set de prueba y validacion para los modelos que use el usuario
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,shuffle=False, test_size = 0.2,random_state=100)
    if modelo=="POLY":
       
        poly_model = LinearRegression()
        poly = PolynomialFeatures(degree=dimension)

        Xpolytrain = poly.fit_transform(X_train)
        Xpolytest = poly.fit_transform(X_test)
        Xfull=poly.fit_transform(X)
        poly_model.fit(Xpolytrain, Y_train)
        y_train_predict = poly_model.predict(Xpolytrain)

        y_test_predict = poly_model.predict(Xpolytest)
        y_test_full = poly_model.predict(Xfull)

        MSE = mean_squared_error(Y_train,y_train_predict)
        print("Entrenamiento: MSE ="+str(MSE))

        MSE = (mean_squared_error(Y_test, y_test_predict))
        print("Pruebas: MSE="+str(MSE))

        return y_test_full

    if modelo=="RF":
        rf = RandomForestRegressor(n_estimators=70)
        rf.fit(X_train, Y_train)
        y_pred = rf.predict(X_train)
        y_pred_rf = rf.predict(X)
        
        print('R2 en el conjunto de entrenamiento RF: ', rf.score(X_train, Y_train))
        print('R2 en el conjunto de prueba RF: ', rf.score(X_test, Y_test))
        return y_pred_rf
    
    if modelo=="SVR":
        svr = SVR(kernel = 'rbf', C = 1e3, gamma = 0.000001)
        svr.fit(X_train, Y_train)
        y_pred_svr = svr.predict(X)
        print('R2 en el conjunto de entrenamiento SVR: ', svr.score(X_train, Y_train))
        print('R2 en el conjunto de prueba SVR: ', svr.score(X_test, Y_test))
        return y_pred_svr

   