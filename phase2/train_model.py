import numby as np 
from sklearn.linear_moel impot LinearRegression 
from sklearn.metrics import mean_absolute_error 

np.random.seed(0)

X=np.ranom.rand(100,5)

y=100 - X.sum(axis=1)*10

model=LinearRegression()
model.fit(X,y)

pred=model.predict(X)

print("Predicted RUL example:", pred[0])
print("MAE:", mean_absolute_error(y,pred))
