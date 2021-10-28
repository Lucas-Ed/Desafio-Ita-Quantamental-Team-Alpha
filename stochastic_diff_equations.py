import backtrader as bt
import numpy as np
from scipy.stats import norm as N

class gBm(bt.Indicator):
    """
        Geometric Brownian Motion model.
                dS_t = μ S_t dt + σ S_t dB_t
        
        Crescimento Exponencial. 
        o modelo tem um fit melhor para growth stocks como AAPL, TSLA, MGLU3, PRIO3 etc i.e.
        para ativos que tem um Crescimento Exponencial
    """
    lines = ('m', 'σ', 'μ', 'sl')
    params = dict(size=2**8+1, 
                  sl_quantil=0.1, 
                  holding_period=5)
    
    def __init__(self):
        self.addminperiod(self.p.size)
        
    def next(self):
        self.S = self.data.close.get(0, size=self.p.size)
        self.estimateParams()
        

        self.lines.m[0] = self.m
        self.lines.σ[0] = self.σ
        self.lines.μ[0] = self.μ
        self.lines.sl[0] = self.q(self.p.sl_quantil, self.p.holding_period)

    
    def estimateParams(self):
        """
        Referência:
        "Estimation of Geometric Brownian Motion Parameters for Oil Price Analysis" C. Jakob et al.
        """
        S = self.S
        X = np.diff(np.log(S), n=1)
        m = X.mean() #mean
        σ = X.std() #standard deviation
        μ = m + ((σ**2)/2) #drift
        n = len(S)

        self.m = m
        self.σ = σ
        self.μ = μ
        self.n = n
    
    def E(self, t):
        """
        Referência:
        Ross, Sheldon M. (2014). "Variations on Brownian Motion".
        Introduction to Probability Models (11th ed.).
        """
        S = self.S
        S0 = S[0]
        μ = self.μ
        return S0*np.exp(μ*t)
    
    def Var(self, t):
        """
        Referência:
        Ross, Sheldon M. (2014). "Variations on Brownian Motion". 
        Introduction to Probability Models (11th ed.).
        """
        S = self.S
        S0 = S[0]
        μ = self.μ
        σ = self.σ
        return (S0**2)*np.exp(2*μ*t)*(np.exp((σ**2)*t) - 1)
    
    def q(self, p, t):
        """
         quantil de St/S0 o qual é definido como:
                q(p) = exp( (μ - σ**2/2)*t + σ*np.sqrt(t)*inv_Φ(p))
         p ∈ (0,1)
        """
        #assert p>0 and p<1
        #assert type(t)==int

        σ = self.σ
        μ = self.μ
        
        mean = (μ - (σ**2/2))*t
        var = σ**2*t
        return np.exp(mean +  np.sqrt(var)*N.ppf(p, 0, 1))
