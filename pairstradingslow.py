#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import time
import numpy as np 
import matplotlib.mlab as mlab
import time
import xlrd
import xlwt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# In[ ]:


def testpairspyslow():
    sh = pd.DataFrame(pd.read_excel('Excel_Workbook.xls',index_col='date'))
    shform=sh['2013-03-01':'2019-12-27'].fillna(method='pad')
    shform.index = pd.to_datetime(shform.index)
    pairs = pairspyslow(shform)
    return pairs


# In[ ]:


def testaccount():
    sh = pd.DataFrame(pd.read_excel('Excel_Workbook.xls',index_col='date'))
    shform=sh['2013-03-01':'2019-12-27'].fillna(method='pad')
    shform.index = pd.to_datetime(shform.index)
    stock1 = shform[600000]
    stock2 = shform[600340]
    account = loopback(stock1,stock2,2000)
    return account


# In[ ]:


def plotaccount(account):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    account['Asset'].plot(label='总资产',style='-')
    plt.title('总资产变化曲线')
    plt.show()


# In[3]:


def pairspyslow(shform):
    #一阶单整
    shpairs = integf(shform)
    Pf = shpairs[0]
    Pfindex = shpairs[1]
    num = len(Pf)
    length = len(Pf[0])
    pairs = []
    for m in range (0 , num-1):
        for n in range(m+1, num):
            #残差平稳性
            values = conintegtest(Pf[m],Pf[n])
            if not(values==0):
                pairs.append([Pfindex[m],Pfindex[n]])
    return pairs


# In[4]:


#判断表格的一阶单整并返回序列和序列名称
def integf(shform):
    length = len(shform.columns.values)
    Pf=[]
    Pfindex=[]
    shform = shform.fillna(method='pad')
    for i in range (0,length-1):
        stock = shform[shform.columns[i]].values
        test = integ(stock)
        if (test==1):
            Pf.append(stock)
            Pfindex.append(shform.columns[i])
    return [Pf,Pfindex]


# In[5]:


#单独判断一阶单整序列
def integ(stock):
    test = fillna(stock)
    length = len(test)
    Plog = np.log(test)
    ret = []
    for i in range (1,length):
        ret.append(Plog[i]-Plog[i-1])
    if slowadf(Plog)==0 and slowadf(ret)==1:
        return 1
    else:
        return 0


# In[6]:


#单只股票空值的填充
def fillna(stock):
    length = len(stock)
    for i in range (1,length):
        if(stock[i]==0 or stock[i]=='nan'):
            stock[i] = stock[i-1]
    return stock


# In[7]:


def slowadf(stock):
    adfctt = adfuller(stock,regression='ctt',autolag=None)
    if adfctt[0]<adfctt[4]['5%']:
        return 1
    else:
        adfct = adfuller(stock,regression='ct')
        if adfct[0]<adfct[4]['5%']:
            return 1
        else:
            adfc = adfuller(stock)
            if adfc[0]<adfc[4]['5%']:
                return 1
            else:
                return 0


# In[8]:


def conintegtest(stock1,stock2):
    stock1 = fillna(stock1)
    stock2 = fillna(stock2)
    log1 = np.log(stock1)
    log2 = np.log(stock2)
    model=sm.OLS(log2,sm.add_constant(log1))
    results = model.fit()
    alpha = results.params[0]
    beta = results.params[1]
    spread = log2 - beta*log1 - alpha
    if (slowadf(spread)==0):
        return 0
    else:
        values={'beta':'','alpha':'','spread':''}
        values['beta'] = beta
        values['alpha'] = alpha
        values['spread'] = spread
        return values


# In[9]:


def loopback(stk1, stk2, money, meancoinf = 1, stdcoinf1 = 0.2, stdcoinf2 = 1.5, stdcoinf3 = 2.5, DR = 1, IR = 0.5, MR = 5):
    if(stk1[0]<stk2[0]):
        stock1 = stk1.values
        stock2 = stk2.values
    else:
        stock1 = stk2.values
        stock2 = stk1.values
    log1 = np.log(stock1)
    log2 = np.log(stock2)
    cash=[money]
    model=sm.OLS(log2,sm.add_constant(log1))
    results = model.fit()
    alpha = results.params[0]
    beta = results.params[1]
    spread = log2 - beta*log1 - alpha
    means = np.mean(spread)
    stds = np.std(spread)
    length = len(stock1)
    flat1 = meancoinf * means + stdcoinf1 * stds
    flat2 = meancoinf * means - stdcoinf1 * stds
    build1 = meancoinf * means + stdcoinf2 * stds
    build2 = meancoinf * means - stdcoinf2 * stds
    force1 = meancoinf * means + stdcoinf3 * stds
    force2 = meancoinf * means - stdcoinf3 * stds
    level = (float('-inf'),force2,build2,flat2,flat1,build1,force1,float('inf'))
    spreadlevel = pd.cut(spread,level,labels=False)-3
    
    signal = tradesignal(spreadlevel)
    position = tradeposit(signal)
    shareY = [0]
    shareX = [0]
    for i in range(1,length):
        g = 1 + beta * stock2[i] / stock1[i]
        g2 = 1 + stock2[i] / (stock1[i] * beta)
        shareX.append(shareX[i-1])
        shareY.append(shareY[i-1])
        cash.append(cash[i-1])
        if position[i-1]==0 and position[i]==-1:
            #卖出x买入y
            L = max(min(cash[i-1]*(DR+IR)/(IR*g2),cash[i-1]*(1+DR)/((MR-1)*g2)),min(cash[i-1]/(IR*g2-DR-IR+1),cash[i-1]/((MR-1)*g2-DR),cash[i-1]))
            S = (g2-1)*L
            shareX[i] = -S
            shareY[i] = L
            cash[i]=cash[i-1]-(shareY[i]*stock2[i]+shareX[i]*stock1[i])
        elif position[i-1]==0 and position[i]==1:
            #买入x卖出y
            L = max(min(cash[i-1]*(DR+IR)/(IR*g),cash[i-1]*(1+DR)/((MR-1)*g)),min(cash[i-1]/(IR*g-DR-IR+1),cash[i-1]/((MR-1)*g-DR),cash[i-1]))
            S = (g-1)*L
            shareX[i] = L
            shareY[i] = -S
            cash[i]=cash[i-1]-(shareY[i]*stock2[i]+shareX[i]*stock1[i])
        elif position[i-1]==1 and position[i]==0:
            shareX[i]=0
            shareY[i]=0
            cash[i]=cash[i-1]+(shareY[i-1]*stock2[i]+shareX[i-1]*stock1[i])    
        elif position[i-1]==-1 and position[i]==0:
            shareX[i]=0
            shareY[i]=0
            cash[i]=cash[i-1]+(shareY[i-1]*stock2[i]+shareX[i-1]*stock1[i])     
    cash = pd.Series(cash,index=stk1.index)
    shareY = pd.Series(shareY,index=stk1.index)
    shareX = pd.Series(shareX,index=stk1.index)
    asset = cash+shareY*stock2+shareX*stock1
    account = pd.DataFrame({'ShareY':shareY, 'ShareX':shareX, 'Cash':cash, 'Asset':asset})
    return account


# In[10]:


def tradesignal(prcLevel):
    length = len(prcLevel)
    signals = np.zeros(length)
    for i in range(1,length):
        if prcLevel[i-1] == 1 and prcLevel[i] == 2:
            signals[i] = -2
        elif prcLevel[i-1] == 1 and prcLevel[i] == 0:
            signals[i] = 2
        elif prcLevel[i-1] == 2 and prcLevel[i] == 3:
            signals[i] = 3   
        elif prcLevel[i-1] == -1 and prcLevel[i] == -2:
            signals[i] = 1
        elif prcLevel[i-1] == -1 and prcLevel[i] == 0:
            signals[i] = -1
        elif prcLevel[i-1] == -2 and prcLevel[i] == -3:
            signals[i] = -3
    return(signals)


# In[11]:


def tradeposit(signal):
    position = [signal[0]]
    ns = len(signal)
    for i in range (1,ns):
        position.append(position[-1])
        if signal[i] == 1:
            position[i] = 1
        elif signal[i] == -2:
            position[i] = -1
        elif signal[i] == -1 and position[i-1]==1:
            position[i] = 0
        elif signal[i] == 2 and position[i-1]==-1:
            position[i] = 0
        elif signal[i] == 3:
            position[i] = 0
        elif signal[i] == -3:
            position[i] = 0
    return np.array(position)


# In[ ]:




