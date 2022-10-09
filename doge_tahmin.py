#%%
from matplotlib.figure import Figure
import pandas as pd
import pandas_datareader.data as pdr 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
plt.style.use('bmh')
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor



df=pd.read_csv('C:\\Users\\user\\Desktop\\DOGE-USD.csv')
df.describe()
df = df[["Close"]]

future_days=30

df["Tahmin"]=df["Close"].shift(-future_days)

X=np.array(df.drop(["Tahmin"],1))[:-future_days]

y = np.array(df["Tahmin"])[:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

x_future = df.drop(["Tahmin"],1)[-future_days:]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

randomForest= RandomForestRegressor(n_estimators=200)
randomForest.fit(X_train,y_train)
accuracy=randomForest.score(X_test,y_test)

randomForest_prediction=randomForest.predict(x_future)

predictions=randomForest_prediction
valid=df[X.shape[0]:]
valid["Tahmin"]=predictions

valuesOfClose = np.array(df["Close"])
valuesOfPrediction = np.array(valid["Tahmin"])
combinated_array = np.append(valuesOfClose,valuesOfPrediction)

extended = pd.DataFrame(columns=["30ekle"])
extended["30ekle"] = combinated_array



plt.figure(figsize = (40,8))
plt.title("DOGECOİN")
plt.xlabel("Günler")
plt.ylabel("Kapanış fiyatı")
plt.xticks([30*x for x in range(0,1860)])
plt.plot(df["Close"].tail(365),color="blue")
plt.plot(extended["30ekle"].tail(30),color="red")
plt.legend(["Close","Pred"])
plt.show()




# %%
