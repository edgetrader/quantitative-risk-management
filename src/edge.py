import numpy as np
import pandas as pd

# Import the plotting library
import matplotlib.pyplot as plt
from matplotlib import patheffects

import seaborn as sns; sns.set()
plt.style.use('ggplot')

from random import shuffle

# !pip install PyPortfolioOpt
# Import the packages 
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

FIG_SIZE = (20,8)

def generateAnalytics(returns):

    analytics = pd.DataFrame([])

    # Risk free rate
    risk_free = 0

    analytics['annual_return'] = returns.describe().loc['mean'] * 252 * 100
    analytics['annual_volatility'] = returns.describe().loc['std'] * np.sqrt(252) * 100
    analytics['sharpe_ratio'] = (analytics['annual_return'] - risk_free)/analytics['annual_volatility']

    return analytics

def prettyAnalytics(returns, tickermap={}):

    analytics = generateAnalytics(returns)
    analytics.columns = ['Annualised Return (%)', 'Annualised Volatility (%)', 'Sharpe Ratio']
    
    if tickermap:
        return analytics.reset_index().replace({'index': tickermap}).set_index('index') \
            .sort_values('Sharpe Ratio', ascending=False).style.bar(color=['lightred', 'lightgreen'], align='zero')
    else:
        return analytics.sort_values('Sharpe Ratio', ascending=False) \
            .style.bar(color=['lightred', 'lightgreen'], align='zero')


def cumulative_returns_plot(returns, tickers):     
    cum_returns = ((1 + returns[tickers]).cumprod()-1) 
    cum_returns.plot() 
    plt.show()


def plotTimeSeries(series, title='', xlabel='', ylabel='', tickermap=''): 
    ax = series.plot(figsize=FIG_SIZE, fontsize=12, linewidth=3, linestyle='-')
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)
    
    title_text_obj = ax.set_title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])
    # pe = patheffects.withSimplePatchShadow(offset = (1, -1), shadow_rgbFace = (1,0,0), calpha = 0.8)

    names = series.columns
    if tickermap:
        names = names.map(tickermap)
        
    ax.legend(names, fontsize = 16)
    plt.show()


def plotReturnDistribution(returns, ticker, binsize=35, title=''):
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.hist(returns[ticker], bins=binsize, color='steelblue', density = True,
             alpha = 0.5, histtype ='stepfilled',edgecolor ='red' )

    title_text_obj = ax.set_title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

    sigma, mu = returns[ticker].std(), returns[ticker].mean() # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)

    count, bins, ignored = ax.hist(s, binsize, density=True, alpha = 0.1)
    ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
            linewidth=1.5, color='r')

    ax.annotate('Skewness: {}\n\nKurtosis: {}'.format(round(returns[ticker].skew(),2),
                                                       round(returns[ticker].kurtosis(),2)),
                 xy=(10,20), xycoords = 'axes points', xytext =(20,360), fontsize=14)

    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    
    plt.show()


def plotCorrelationMatrixWithCluster(matrix, title= ''):

    fig = sns.clustermap(matrix, row_cluster=True, col_cluster=True, figsize=(10,10))  

    title_text_obj = plt.title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])
    
    plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    plt.show()


def plotCorrelationMatrix(matrix, type=1, title= ''):

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    title_text_obj = ax.set_title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

    if type == 1:
        sns.heatmap(matrix, annot=True, linewidths=.5)
    elif type == 2:
        sns.heatmap(matrix, annot=True, cmap="YlGnBu", linewidths=0.3, annot_kws={"size": 16})
    else:
        sns.heatmap(matrix, annot=True, cmap='coolwarm', linewidths=0.3, annot_kws={"size": 16})

    # Plot aesthetics
    plt.xticks(rotation=90)
    plt.yticks(rotation=0) 
    b, t = plt.ylim() # discover the values for bottom and top
#     b += 0.5 # Add 0.5 to the bottom
#     t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.show()
    

def compareDistribution(series, tickers, title='', tickermap={}):    
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    title_text_obj = ax.set_title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

    for ticker in tickers:
        sns.distplot(series[ticker])

    names = series.columns
    if tickermap:
        names = names.map(tickermap)

    ax.legend(names, fontsize = 12)
    ax.set_xlabel('Returns Distribution', fontsize= 14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.show()


def generateWgts(num):
    wgts = np.random.random(num)
    wgts /= wgts.sum()
    return wgts


def drawPie(table, column, tickers, title = ''):
    fig = plt.figure(figsize=FIG_SIZE)
   
    # Pie chart
    labels = tickers
    sizes = table[column]

    #colors
    slices = [1,2,3] * 4 + [20, 25, 30] * 2
    shuffle(slices)
    cmap = plt.cm.prism
    colors = cmap(np.linspace(0., 1., len(slices)))

    #explosion
    explode = 0.05 * np.ones(len(tickers))

    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', 
             startangle=90, pctdistance=0.85, explode = explode, textprops={'fontsize': 16})

    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.title(title, fontsize = 16)
    plt.tight_layout()
    plt.show()

def simAllocation(num, tickers): 
    wgts = {}
    for i in range(0,num):
        wgts['Simulated_' + str(i)] = generateWgts(len(tickers))
    wgts['Balanced'] = 1/len(tickers) * np.ones(len(tickers))
    
    allocation = pd.DataFrame(data = wgts, index = tickers)    
    return allocation


def plotEfficientFrontier(returns, tickers, numofsim=200, title='Simulated Efficient Frontier'):

    allocation = simAllocation(numofsim, tickers)
    port_returns = returns.dot(allocation)

    analytics = generateAnalytics(port_returns)
    
    rets = np.array(analytics.annual_return)/100
    vols = np.array(analytics.annual_volatility)/100
    sr = np.array(analytics.sharpe_ratio)

    # the charts
    fig8 = plt.figure(figsize = FIG_SIZE)
    plt.subplots_adjust(wspace=.5)
    plt.subplot()

    plt.scatter(vols, rets, c = sr, marker = 'o',cmap='coolwarm')
    plt.grid(True)
    plt.xlabel('annual volatility')
    plt.ylabel('annual return')
    plt.colorbar(label = 'Sharpe Ratio')

    title_text_obj = plt.title(title, fontsize = 18, verticalalignment = 'bottom')
    title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

    plt.show()


def genEFdata(ef, vsteps = 0.001, rf=0.00):
    
    RISK_FREE_RATE = rf

    results = pd.DataFrame([])

    min_vol = ef.min_volatility()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate = RISK_FREE_RATE)
    perf_dict = {}
    perf_dict['index'] = 'min_vol'
    perf_dict['annual_return'] = perf[0]
    perf_dict['annual_volatility'] = perf[1]
    perf_dict['sharpe_ratio'] = perf[2]
    results = results.append(perf_dict, ignore_index=True)
    lower_bound = perf[1]

    max_sharpe = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    perf = ef.portfolio_performance(verbose=False, risk_free_rate = RISK_FREE_RATE)
    perf_dict = {}
    perf_dict['index'] = 'max_sharpe'
    perf_dict['annual_return'] = perf[0]
    perf_dict['annual_volatility'] = perf[1]
    perf_dict['sharpe_ratio'] = perf[2]
    results = results.append(perf_dict, ignore_index=True)
    upper_bound = perf[1] * 2

    count = 0
    
    lower_bound = 0
    for vol in np.arange(lower_bound, upper_bound, vsteps):
        try:
            wgt = ef.efficient_risk(vol, risk_free_rate=RISK_FREE_RATE)
            perf = ef.portfolio_performance(verbose=False, risk_free_rate = RISK_FREE_RATE)

            count += 1
            perf_dict = {}
            perf_dict['index'] = 'calc_' + str(count)
            perf_dict['annual_return'] = perf[0]
            perf_dict['annual_volatility'] = perf[1]
            perf_dict['sharpe_ratio'] = perf[2]

            results = results.append(perf_dict, ignore_index=True)
        except:
            continue
            
    return results.set_index('index')


def plotEF(ef, vsteps=0.001, rf=0.0):
    
    analytics = genEFdata(ef, vsteps=vsteps, rf=rf)    

    minvol_rets = analytics.loc['min_vol'].annual_return
    minvol_vols = analytics.loc['min_vol'].annual_volatility
    maxsr_rets = analytics.loc['max_sharpe'].annual_return
    minsr_vols = analytics.loc['max_sharpe'].annual_volatility
    
    rets = np.array(analytics.annual_return)
    vols = np.array(analytics.annual_volatility)
    sr = np.array(analytics.sharpe_ratio)

    # the charts
    fig8 = plt.figure(figsize = FIG_SIZE)
    plt.subplots_adjust(wspace=.5)
    plt.subplot()

    plt.scatter(vols, rets, c = sr, marker = 'o', cmap='coolwarm')
    plt.scatter(minvol_vols, minvol_rets, c = 'r', s = 300, marker = '*', label='Minimum volatility')
    plt.scatter(minsr_vols, maxsr_rets, c = 'g', s = 300, marker = '*', label='Maximum sharpe ratio')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.title('Simulated Efficient Frontier')
    plt.legend(labelspacing=0.8)
    plt.show()