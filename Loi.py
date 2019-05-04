import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sps


class LoiProb :
  
 def exp(self,lamda,nbr=100):
  data = np.random.exponential(lamda, size=nbr)
  hist,edges = np.histogram(data,bins="auto",density=True )
  x = edges[:-1]+np.diff(edges)/2.
  plt.scatter(x,hist)


  func = lambda x,beta: 1./beta*np.exp(-x/beta)
  popt, pcov = curve_fit(f=func, xdata=x, ydata=hist) 
  print(popt)
  xx = np.linspace(0, x.max(), nbr)
  plt.plot(xx, func(xx,*popt), ls="--", color="k", label="fit, $beta = ${}".format(popt))
  plt.legend()
  plt.show()

 def poiss(self,lamda,nbr=100):
  s = np.random.poisson(lamda, nbr)
  plt.hist(s, 14, density=True)
  plt.show()

 def normal(self,mu,sigma,nbr=100):

  s = np.random.normal(mu, sigma, nbr)
  ount, bins, ignored = plt.hist(s, 30, density=True)
  plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),      linewidth=2, color='r')
  plt.show()

 def unif(self,a,b,nbr=100):
   s = np.random.uniform(a,b,nbr)
   count, bins, ignored = plt.hist(s, 15, density=True,histtype='stepfilled')
   plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
   plt.show()
  
 
 def gamma(self,n,lamda,nbr=100):
    shape, scale = 2., 2.
    s = np.random.gamma(shape, scale, nbr)
    count, bins, ignored = plt.hist(s, 50, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale)  /  (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')
    plt.show()

