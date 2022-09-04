#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import time
import numpy as np 
import matplotlib.mlab as mlab
import time
import xlrd
import xlwt


# In[ ]:


def testpairspyquick():
    sh = pd.DataFrame(pd.read_excel('Excel_Workbook.xls',index_col='date'))
    shform=sh['2013-03-01':'2019-12-27'].fillna(method='pad')
    shform.index = pd.to_datetime(shform.index)
    pairs2 = pairspyquick(shform)
    return pairs2


# In[1]:


def pairspyquick(shform):
    shpairs = integfq(shform)
    Pf = shpairs[0]
    Pfindex = shpairs[1]
    num = len(Pf)
    length = len(Pf[0])
    pairs = []
    for m in range (0 , num-1):
        for n in range(m+1, num):
            #残差平稳性
            values = conintegtestq(Pf[m],Pf[n])
            if not(values==0):
                pairs.append([Pfindex[m],Pfindex[n]])
    return pairs


# In[2]:


#判断表格的一阶单整并返回序列和序列名称
def integfq(shform):
    length = len(shform.columns.values)
    Pf=[]
    Pfindex=[]
    shform = shform.fillna(method='pad')
    for i in range (0,length-1):
        stock = shform[shform.columns[i]].values
        test = integq(stock)
        if (test==1):
            Pf.append(stock)
            Pfindex.append(shform.columns[i])
    return [Pf,Pfindex]


# In[3]:


#单独判断一阶单整序列
def integq(stock):
    test = fillna(stock)
    length = len(test)
    Plog = np.log(test)
    ret = []
    for i in range (1,length):
        ret.append(Plog[i]-Plog[i-1])
    if myadf(Plog)==0 and myadf(ret)==1:
        return 1
    else:
        return 0


# In[4]:


#单只股票空值的填充
def fillna(stock):
    length = len(stock)
    for i in range (1,length):
        if(stock[i]==0 or stock[i]=='nan'):
            stock[i] = stock[i-1]
    return stock


# In[5]:


#对两只股票进行检验
def conintegtestq(stock1,stock2):
    stock1 = fillna(stock1)
    stock2 = fillna(stock2)
    log1 = np.log(stock1)
    log2 = np.log(stock2)
    model = myols(log1,log2)
    if (model[0]==0):
        return 0
    else:
        spread = model[1]
        if (myadf(spread)==0):
            return 0
        else:
            values={'beta':'','alpha':'','spread':''}
            values['beta'] = model[2]
            values['alpha'] = model[3]
            values['spread'] = spread
            return values


# In[6]:


def myols(stock1,stock2):
    numerator = 0
    denominator = 0
    xsum = 0
    ysum = 0
    length = len(stock1)
    for i in range (0,length):
        xsum = xsum + stock1[i]
        ysum = ysum + stock2[i]
    xmean = xsum/length
    ymean = ysum/length
    for i in range (0,length):
        numerator = numerator + (stock1[i]-xmean)*(stock2[i]-ymean)
        denominator = denominator + (stock1[i]-xmean)*(stock1[i]-xmean)
    beta = numerator/denominator

    #回归分析：计算截距alpha
    alpha = ymean - beta*xmean

    #回归分析：计算残差epsilon
    epsilon = stock2-beta*stock1-alpha
    
    #计算残差平方和与解释变量平方和
    SSR=0
    temp=0
    for i in range (0,length):
        SSR = SSR + epsilon[i]**2
        temp = temp + stock1[i]**2
        
    tb = beta/((SSR/denominator)**0.5)
    ta = alpha/((SSR*temp/length/denominator)**0.5)
    return [1,epsilon,beta,alpha]


# In[15]:


def myadf(stock1):
    if(type(stock1).__name__=='Series'):
        stock = stock1.values    
    else:
        stock = stock1
    T = len(stock)
    diff = np.array(stock)[1:]-np.array(stock)[:-1]
    t=[-1.95,-2.86,-3.41]
    #最大窗口
    pmax = 12*int((T/100)**(1/4))
    Y = np.mat(np.array(diff[(pmax-1):]))
    n = Y.shape[1]
    y = np.mat(np.array(stock[(pmax-1):-1]))
    X = np.mat([1.0]*n)

    #第三方程
    times = []
    for i in range(0,T-pmax):
        times.append(i)
    X3 = np.insert(X,1,np.array(times),0)
    X3 = np.insert(X3,2,np.array(y),0)
    #第三方程最佳窗口
    bestp3 = pmax
    for i in range(0,bestp3-1):
        X3=np.insert(X3,3+i,np.array(diff[(pmax-i-2):(-i-1)]),0)
    X3 = X3.T
    X3inv = quickinverse(X3.T*X3)
    B3 = (X3inv)*(X3.T*Y.T)
    E3 = X3*B3-Y.T
    SSR3 = 0
    for i in range (T-pmax):
        SSR3 = SSR3 + E3[i,0]**2
    t3gamma = float(B3[2]/((X3inv[2,2]*SSR3/(T-pmax-bestp3-2))**0.5))
    t3t = float(B3[1]/((X3inv[1,1]*SSR3/(T-pmax-bestp3-2))**0.5))
    if (t3gamma<t[2]):
        #为平稳序列
        return 1
    else:

    #第二方程
        X2 = np.insert(X,1,np.array(y),0)
        #第二方程最佳窗口
        bestp2 = pmax
        for i in range(0,bestp2-1):
            X2 = np.insert(X2,2+i,np.array(diff[(pmax-i-2):(-i-1)]),0)
        X2 = X2.T
        X2inv = quickinverse(X2.T*X2)
        B2 = (X2inv)*(X2.T*Y.T)
        E2 = X2*B2-Y.T
        SSR2 = 0
        for i in range (T-pmax):
            SSR2 = SSR2 + E2[i,0]**2
        t2gamma = float(B2[1]/((X2inv[1,1]*SSR2/(T-pmax-bestp2-2))**0.5))
        t2a = float(B2[0]/((X2inv[0,0]*SSR3/(T-pmax-bestp2-2))**0.5))
        if (t2gamma<t[1]):
            #为平稳序列
            return 1
        else:
            #如果a也显著不为零，则不平稳
            X1 = X
            #第一方程
            #第一方程最佳窗口
            bestp1 = pmax
            for i in range(0,bestp1-1):
                X1=np.insert(X1,1+i,np.array(diff[(pmax-i-2):(-i-1)]),0)
            X1 = X1.T
            X1inv = quickinverse(X1.T*X1)
            B1 = (X1inv)*(X1.T*Y.T)
            E1 = X1*B1-Y.T
            SSR1 = 0
            for i in range (T-pmax):
                SSR1 = SSR1 + E1[i,0]**2
            t1gamma = float(B1[0]/((X1inv[0,0]*SSR1/(T-pmax-bestp1-2))**0.5))
            if(t1gamma<t[0]):
                #为平稳序列
                return 1
            else:
                return 0


# In[8]:


def adfbestwindow(pmax,origin,deltay,diff):
    T = origin.shape[1]
    SC=[]
    Y=deltay.T
    adftype = origin.shape[0]
    #i+1为滞后阶数
    for i in range(0,pmax-1):
        origin=np.insert(origin,adftype+i,np.array(diff[(pmax-i-2):(-i-1)]),0)
        X=origin.T
        B = (quickinverse(X.T*X))*(X.T*Y)
        E = X * B - Y
        aictemp = 0
        for j in range(0,T):
            aictemp = aictemp + E[j,0]**2
        #SC.append((i+1+1)*np.log(T)+T*np.log(aictemp/T))
        SC.append(2*(i+1)+T*np.log(aictemp/T))
    bestwindow = 0
    for i in range (0,len(SC)):
        if SC[i]<SC[bestwindow]:
            bestwindow = i
    return (i+1)


# In[10]:


def quickinverse(A):
    dim = A.shape[0]
    E = np.mat(np.eye(dim))
    L = np.mat(np.eye(dim))
    U = A.copy()
    for i in range(dim):
        if abs(A[i,i]) < 1e-8:
            print("主对角线有零值！")
            sys.exit()
        L[i+1:,i] = U[i+1:,i] / U[i,i]

        E[i+1:,:] = E[i+1:,:] - L[i+1:,i]*E[i,:]
        
        U[i+1:,:] = U[i+1:,:] - L[i+1:,i]*U[i,:]

    Uinv = np.mat(np.eye(dim))
    for i in range(dim-1,-1,-1):
        # 对角元单位化
        E[i,:] = E[i,:]/U[i,i]
        Uinv[i,:] = Uinv[i,:]/U[i,i]
        U[i,:] = U[i,:]/U[i,i]

        E[:i,:] = E[:i,:] - U[:i,i]*E[i,:]
        Uinv[:i,:] = Uinv[:i,:] - U[:i,i]*Uinv[i,:]
        U[:i,:] = U[:i,:] - U[:i,i]*U[i,:] # r_j = m_ji - r_j*r_i

    Linv = np.mat(np.eye(dim))
    for i in range(dim):
        Linv[i+1:,:] = Linv[i+1:,:] - L[i+1:,i]*Linv[i,:]
        L[i+1:,:] = L[i+1:,:] - L[i+1:,i]*U[i,:]

    return Uinv*Linv


# In[11]:


def tvalue(length):
    if length<25:
        return [-1.95,-3.0,-3.6]
    elif length<50:
        return [-1.95,-2.93,-3.5]
    elif length<100:
        return [-1.95,-2.89,-3.45]
    elif length<250:
        return [-1.95,-2.88,-3.43]
    elif length<500:
        return [-1.95,-2.87,-3.42]
    else:
        return [-1.95,-2.86,-3.41]


# In[ ]:




