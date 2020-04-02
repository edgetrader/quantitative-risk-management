from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.objective_functions import negative_cvar
from pypfopt import expected_returns
from scipy.stats import norm, skewnorm, t, gaussian_kde 
from scipy import integrate
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

# Import the plotting library
import matplotlib.pyplot as plt
from matplotlib import patheffects

import seaborn as sns; sns.set()
plt.style.use('ggplot')

## Efficient Frontier Class
class ef():
    
    def __init__(self, prices, risk_free=0.00):
        self.prices = prices
        self.assets = prices.columns
        self.num_of_assets = len(prices.columns)
        self.e_returns = expected_returns.mean_historical_return(prices)
        self.returns = prices.pct_change().dropna()
        self.cum_returns = ((1 + self.returns).cumprod()-1) 
        self.correlation = self.returns.corr('pearson')
        self.covariance = CovarianceShrinkage(prices).ledoit_wolf()
        self.rf = risk_free

    def getPrices(self):
        return self.prices
    
    def min_CVaR(self):
        prices = self.prices
        returns = self.returns
        cov = self.covariance
        
        ef = EfficientFrontier(None, cov)
        optimal_weights = ef.custom_objective(negative_cvar, returns)
        return optimal_weights, None
    
    def max_sharpe(self):
        prices = self.prices
        returns = self.returns
        rf = self.rf
        
        e_returns = self.e_returns
        cov = self.covariance
        ef = EfficientFrontier(e_returns, cov)

        optimal_weights = ef.max_sharpe(risk_free_rate=rf)
        perf = ef.portfolio_performance(verbose=False, risk_free_rate = rf)  
        return optimal_weights, perf

    def min_volatility(self):
        prices = self.prices
        returns = self.returns
        rf = self.rf
        
        e_returns = self.e_returns
        cov = self.covariance
        ef = EfficientFrontier(e_returns, cov)

        optimal_weights = ef.min_volatility()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate = rf)  
        return optimal_weights, perf

## Portfolio Class
class portfolio():

    def __init__(self, weights, prices, risk_free=0.00):
        self.weights = weights
        self.prices = prices
        self.assets = prices.columns
        self.num_of_assets = len(prices.columns)
        self.returns = prices.pct_change().dropna()
        self.e_returns = expected_returns.mean_historical_return(prices)
        self.allocation = pd.DataFrame.from_dict(self.weights, orient='index', columns=['portfolio'])
        self.port_returns = self.returns.dot(self.allocation)
        self.port_cum_returns = ((1 + self.port_returns).cumprod()-1) 
        self.covariance = CovarianceShrinkage(prices).ledoit_wolf()
        self.rf = risk_free

        
    # Standard Normal Distribution Risk Measures
    def calc_risk(self, confidence=0.95):
        port_returns = self.returns.dot(self.allocation)
        losses = -port_returns.iloc[:,0]

        params = norm.fit(losses)
        VaR = norm.ppf(confidence, *params)

        tail_loss = norm.expect(lambda x: x, loc = params[0], scale = params[1], lb = VaR)
        CVaR = (1 / (1 - confidence)) * tail_loss
        return losses, VaR, CVaR

    
    # Standard Normal Distribution Risk Measures - Plots
    def plot_risk(self, confidence=0.95):

        losses, VaR, CVaR = self.calc_risk(confidence)
        losses = -port_returns.iloc[:,0]

        # Fit the portfolio loss data to the skew-normal distribution
        params = norm.fit(losses)
         
        confidence_level = confidence*100
        
        # Plot the normal distribution histogram and add lines for the VaR and CVaR
        plt.title('Portfolio Risk')
        plt.hist(norm.rvs(size = 100000, loc = params[0], scale = params[1]), bins = 100)
        plt.axvline(x = VaR, c='r', 
                    label = "VaR, {}% confidence level - {:.2f}%".format(confidence_level, VaR*100))
        plt.axvline(x = CVaR, c='g', 
                    label = "CVaR, worst {}% of outcomes - {:.2f}%".format(100-confidence_level, CVaR*100))
        plt.legend()
        plt.show()

        
    ## Other Risk Measures
    # Standard Normal Distribution Monte Carlo Simulation with CovarianceShrinkage
    def calc_risk_montecarlo(self, confidence=0.95, N=1000, total_steps=1, random_state=3567):
        e_returns = self.returns.mean().to_frame() ## daily expected returns
        e_cov = self.returns.cov()
        # e_cov = self.covariance
        num_of_assets = self.num_of_assets

        # Initialize daily cumulative loss for the assets, across N runs
        daily_loss = np.zeros((num_of_assets,N))

        # Create the Monte Carlo simulated runs for each asset with correlated randomness
        # N: number of runs
        # total_steps: number of minutes per day
        for n in tqdm(range(N)):
            # Compute simulated path of length total_steps for correlated returns
            correlated_randomness = e_cov @ norm.rvs(size = (num_of_assets, total_steps))
            # Adjust simulated path by total_steps and mean of portfolio losses
            steps = 1/total_steps
            minute_losses = e_returns * steps + correlated_randomness * np.sqrt(steps)
            daily_loss[:, n] = list(minute_losses)
            
        # Calculate portfolio losses
        losses = pd.DataFrame(daily_loss).T
        losses.columns = self.assets
        losses = -losses.dot(self.allocation).iloc[:,0]

        params = norm.fit(losses)
        VaR = norm.ppf(confidence, *params)
        
        tail_loss = norm.expect(lambda x: x, loc = params[0], scale = params[1], lb = VaR)
        CVaR = (1 / (1 - confidence)) * tail_loss
        
        return losses, VaR, CVaR
       
        
    # Student t Distribution Risk Measures
    def calc_risk_t(self, confidence=0.95):
        port_returns = self.returns.dot(self.allocation)
        losses = -port_returns.iloc[:,0]

        params = t.fit(losses)
        VaR = t.ppf(confidence, *params)

        tail_loss = t.expect(lambda y: y, args = (params[0],), loc = params[1], scale = params[2], lb = VaR)
        CVaR = (1 / (1 - confidence)) * tail_loss
        
        return losses, VaR, CVaR
    
    
    # Skew Normal Distribution Risk Measures
    def calc_risk_skewnorm(self, confidence=0.95):
        port_returns = self.returns.dot(self.allocation)
        losses = -port_returns.iloc[:,0]

        params = skewnorm.fit(losses)
        VaR = skewnorm.ppf(confidence, *params)

        tail_loss = skewnorm.expect(lambda y: y, args = (params[0],), loc = params[1], scale = params[2], lb = VaR)
        CVaR = (1 / (1 - confidence)) * tail_loss
        
        return losses, VaR, CVaR


    # Gaussian Kernel Distribution Risk Measures
    def calc_risk_gaussiankernel(self, confidence=0.95):
        losses = -self.port_returns.iloc[:,0]

        kde = gaussian_kde(losses)

        sample = kde.resample(size = 100000) 
        VaR = np.quantile(sample, confidence) 

        def expected_shortfall(dist, cutoff = -np.inf, confidence=confidence):
            fn = lambda x: x * dist.pdf(x)
            out = integrate.quad(fn, a = cutoff, b = np.inf)[0]
            out = (1 / (1 - confidence)) * out
            return out

        CVaR = expected_shortfall(kde, VaR)
        
        return losses, VaR, CVaR


    # Student t or Standard Normal Distribution - Rolling Time Series Risk Measures
    def calc_rolling_VaR(self, window=30, confidence=0.95, student=True):
        losses, VaR, CVaR = self.calc_risk(confidence)

        # Create rolling window parameter list
        mu = losses.rolling(window).mean()
        sigma = losses.rolling(window).std()

        if student:
            df = window - 1
            # df being the degree of freedom = n-1 for student t-distribution
            rolling_parameters = [(df, mu[i], s) for i,s in enumerate(sigma)]
            VaR = np.array( [ t.ppf(confidence, *params) for params in rolling_parameters ] )
        else:
            rolling_parameters = [(mu[i], s) for i,s in enumerate(sigma)]
            VaR = np.array( [ norm.ppf(confidence, *params) for params in rolling_parameters ] )
            
        return losses, VaR