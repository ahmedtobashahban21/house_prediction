import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_absolute_error
###### Regression algorithms 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 


######## loading the data  
load_data = pd.read_csv('E:\\courses\\machine_learning\\training_in_github\\house_prediction\\house.csv')
#### show the details of data 
print(load_data.describe())
#### show the shape of data 
print(load_data.shape) 
## show the non numerical number 
print(load_data['date'])  ####### 2015 or 2014 
sns.countplot(load_data['bedrooms']) ### show the number of rooms

#to conver all cels to numerical numbers
i = 0 
listing=[]
for value in load_data['date']:
    if(( (value[0]+value[1]+value[2]+value[3] )==8)):
       listing.append(1)
    else:
        listing.append(0)


for val in load_data['date']:
    load_data['date'][i]=listing[i]
    i+=1

print(load_data['date'])   ## show the data 
plt.scatter(load_data['date'] , load_data['price']) ## show the relationshipe bitween date and price
#### droping the column 
load_data = load_data.drop(['date'] , axis=1)
print(load_data.shape)      #### show the new shape 
##########  concatenate couples of columns 
plt.scatter(load_data['sqft_living'] , load_data['price'])
plt.scatter(load_data['waterfront'] , load_data['price'])
load_data = load_data.drop(['waterfront']  , axis=1)
print(load_data.shape)   ##dropoing the column from data 
plt.scatter(load_data['zipcode'] , load_data['price']) 
load_data = load_data.drop(['zipcode']  , axis=1)
print(load_data.shape)
plt.scatter(load_data['floors'] , load_data['price'])
## droping the column 
load_data = load_data.drop(['floors'] , axis=1) 
print(load_data.shape)




x = load_data.drop(['id' , 'price'] , axis=1) 
y =load_data['price']
### cleaning data 
clean=SimpleImputer(missing_values=(np.nan))
cleaning_data=clean.fit_transform(x)
## scalling for data 
scal_data=StandardScaler() 
X = scal_data.fit_transform(cleaning_data)


 
## spliting data 
X_train , X_test , y_train  ,y_test = train_test_split(X , y  , test_size=0.33 , random_state=33 , shuffle=True)

#using regression algoritm
LR = LinearRegression() 
LR.fit(X_train , y_train)
y_predict = LR.predict(X_test)
print('LR tarin score :' , LR.score(X_train , y_train))
print("LR test score :" , LR.score(X_test , y_test))
print("LR mean absolute error : " , mean_absolute_error(y_test , y_predict))
print('************************')


#using tree 
DTR = DecisionTreeRegressor()
DTR.fit(X_train , y_train) 
y_pred = DTR.predict(X_test) 
print('DTR train score :' , DTR.score(X_train , y_train))
print('DTR test score :' , DTR.score(X_test  , y_test)) 
print('DTR mean absolute error :' , mean_absolute_error(y_test , y_pred))
print('************************')
#using random forest 
RF = RandomForestRegressor(n_estimators=200 , max_depth=5 , min_samples_split=2)
RF.fit(X_train , y_train) 
y_pre= RF.predict(X_test) 
print("RF train score :" , RF.score(X_train , y_train))
print("RF test score :" , RF.score(X_test , y_test) )
print("RF mean absolute error :" , mean_absolute_error(y_test ,y_pre))
print('************************')








