import numpy as np 
import pandas as pd 
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import mlflow

def evaluation_metrics(y_test,y_pred):
    rmse = mean_squared_error(y_test,y_pred,squared = False)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return rmse,mae,r2

if __name__ == "__main__":
    data = pd.read_csv(r'F:\GUVI_DATA_SCIENCE\Charts\DataSets\Wine_Quality_Data .csv')

    X = data.drop('quality',axis = 1)
    y = data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    with mlflow.start_run():
        alpha = 0.7
        l1_ratio = 0.4

        lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        lr.fit(X_train,y_train)

        predictions = lr.predict(X_test)

        rmse, mae, r2 = evaluation_metrics(y_test,predictions)
        print(f"RMSE : {rmse} | MAE : {mae} | r2_score : {r2}")

        # For logging the parameters
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio) 

        # For logging the evaluation metrics
        mlflow.log_metric("RMSE",rmse)
        mlflow.log_metric("MAE",mae)
        mlflow.log_metric("r2",r2)

        # For logging models
        mlflow.sklearn.log_model(lr,"Model")

        # To view in a User Interface
        'mlflow ui in CMD' 