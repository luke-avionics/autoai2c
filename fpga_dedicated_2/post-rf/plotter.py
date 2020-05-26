import matplotlib.pyplot as plt
import math
fname1="dw_std1.svg"
fname2="dw_std2.svg"
x=[64,96,128,192,288,576,1052]
y_dw1=[1/47241,1/70267,1/84675,1/125835,1/181391,1/348071,1/625861]
y_std1=[1/1709/64,1/1709/96,1/1709/128,1/1709/192,1/1709/288,1/1709/576,1/1709/1052]
log_y_dw1=[]
for i in y_dw1:
    log_y_dw1.append(math.log10(i))
log_y_std1=[]
for i in y_std1:
    log_y_std1.append(math.log10(i))
improvments1=[]
for i in range(len(y_dw1)):
    improvments1.append(y_dw1[i]/y_std1[i])
print(improvments1)
y_dw2=[1/159523,1/236931,1/309251,1/459187,1/681443,1/1348163,1/2459363]
y_std2=[1/7029/64,1/7029/96,1/7029/128,1/7029/192,1/7029/288,1/7029/576,1/7029/1052]
log_y_dw2=[]
for i in y_dw2:
    log_y_dw2.append(math.log10(i))
log_y_std2=[]
for i in y_std2:
    log_y_std2.append(math.log10(i))
improvments2=[]
for i in range(len(y_dw2)):
    improvments2.append(y_dw2[i]/y_std2[i])
print(improvments2)


plt.figure(0)
plt.plot(x, log_y_dw1, '-o',color="darkorange",fillstyle="none")
plt.plot(x, log_y_std1, '-o',color="darkblue",fillstyle="none")
plt.savefig(fname1)
plt.figure(1)
plt.plot(x, log_y_dw2, '-o',color="darkorange",fillstyle="none")
plt.plot(x, log_y_std2, '-o',color="darkblue",fillstyle="none")
plt.savefig(fname2)