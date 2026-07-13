import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score   # type: ignore


### THIS IS FOR BETTER UNDERSTANDING WITH CLOSED FORM AS WELL...

x_actual = np.array([1, 2, 3, 4, 5]) 
y_actual = np.array([2, 4, 5, 4, 5])

# # we need to find m  and c
# #          m = (x-mean_x)(y-mean_y)/(x-mean_x)**2
# #          c = mean_y - m * mean_x

# finding required solu for m and c
x_mean = np.mean(x_actual)
y_mean = np.mean(y_actual)

numer = 0
denom = 0

n = len(x_actual)

for i in range(n):
        numer += (x_actual[i]-x_mean)*(y_actual[i]-y_mean)
        denom += (np.square(x_actual[i]-x_mean))

# here m is slope and c is y-interceptor
m = numer / denom
c = y_mean - (m * x_mean) 


x_min = min(x_actual) 
x_max = max(x_actual) 

x = np.linspace(x_min , x_max ) # feature
y_pred = (m *x) + c # predicted value

# ploting of values with m and c 
plt.plot(x, y_pred)
plt.title(" This is linear regression with closed form..")
plt.xlabel(" thsi is b values ")
plt.ylabel("this is a values ")
plt.scatter(x_actual ,y_actual)
plt.grid(alpha = 0.50)
plt.show()

# this is exactly how closed form works even in sklearn 
# Also same for the even n number of features 

y_pred1 = (m*x_actual) + c

# R2 tells you how well is your line fits ...

#       r2 = ss_reg / ss_tot  or = 1 - (ss_res)/(ss_tot)

#                         where, 
#                               ss_reg =  sum((y_pred - y_mean)**2)
#                               ss_tot = sum(( y - y_mean)**2)
#                               ss_res = sum((y - y_pred )**2)
#                               ss_tot = ss_reg + ss_res



ss_reg =  np.sum((y_pred1 - y_mean)**2)
ss_tot = np.sum(( y_actual - y_mean)**2)
ss_res = np.sum((y_actual - y_pred1 )**2)


#  both are same 
r2_score1 = ss_reg / ss_tot
r2_score2 = 1 - (ss_res/ss_tot)

# with sklearn method just to conform
r2_score3 = r2_score(y_actual , y_pred1)
Mean_Squared_Error = mean_squared_error(y_actual , y_pred1)
Mean_Absolute_Error = mean_absolute_error(y_actual , y_pred1)