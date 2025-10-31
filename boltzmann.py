import gmpy2
import numpy as np
from scipy.optimize import brentq

def generating_series_T(N):
    """
    Computes the N first terms of the generating series of
    rooted unlabelled trees.
    """
    divs = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1, i):
            divs[j].append(i)
    res = [gmpy2.mpz(0)]*(N+1)
    res2 = [gmpy2.mpz(0)]*(N+1)
    res[1] = gmpy2.mpz(1)
    res2[1] = gmpy2.mpz(1)
    for n in range(1, N):
        res[n+1] = sum([res[n-k+1] * res2[k] for k in range(1, n+1)]) // gmpy2.mpz(n)
        res2[n+1] = sum([gmpy2.mpz(d) * res[d] for d in divs[n+1]])
    return res

# T5000 = generating_series_T(5000)
# should take around 7 seconds.

T353 = generating_series_T(353)
T = T353[:350]
Tprime = [i*T353[i] for i in range(1, 351)]
Tsecond = [(i-1)*i*T353[i] for i in range(2, 352)]
Tthird =  [(i-2)*(i-1)*i*T353[i] for i in range(3, 353)]

rho = 2.955765285651995
c = 0.4399240125710253
rhoinv = 0.33832185689920768

def _eval_T_small_values(x):
    res = gmpy2.mpfr(0)
    for coeff in reversed(T):
          res = res * gmpy2.mpfr(x) + coeff
    return float(res)

def eval_T(x):
    """
    Computes the value of T(x), for x<rhoinv. 

    Note that T(rhoinv) = 1.
    """
    if x < 0.3:
        return _eval_T_small_values(x)
    else:
        C = x * np.exp(sum([_eval_T_small_values(x**k) / k for k in range(2, 50)]))
        newton = ( lambda t : C * np.exp(t) * (1 - t)/(1 - C * np.exp(t)) )
        p = 0.5 
        for _ in range(100):
            p = newton(p)
        return float(p)          

def eval_Tprime(x):
    """
    Computes the value of T'(x), for x<rhoinv**2. 
    """
    res = gmpy2.mpfr(0)
    for coeff in reversed(Tprime):
          res = res * gmpy2.mpfr(x) + coeff
    return float(res)

def eval_Tsecond(x):
    """
    Computes the value of T''(x), for x<rhoinv**2.
    """    
    res = gmpy2.mpfr(0)
    for coeff in reversed(Tsecond):
          res = res * gmpy2.mpfr(x) + coeff
    return float(res)

def eval_Tthird(x):
    """
    Computes the value of T'''(x), for x<rhoinv**2.
    """    
    res = gmpy2.mpfr(0)
    for coeff in reversed(Tthird):
          res = res * gmpy2.mpfr(x) + coeff
    return float(res)


def expectation_size(x, pointed=False):
    """
    Computes the expected size of the random tree with Boltzmann parameter x.
    """
    A = 1 - eval_T(x)
    B = (1 + sum([x**k * eval_Tprime(x**k) for k in range(2, 50)]))
    if pointed:
        C = sum([k * (x**k) * eval_Tprime(x**k) 
                 + k * (x**(2*k)) * eval_Tsecond(x**k) 
                 for k in range(2, 50)])
        return B/(A**2) + C/B
    else:
        return B/A

def variance_size(x, pointed=False):
    """
    Computes the variance of the size of the random tree with Boltzmann parameter x.
    """
    A = 1 - eval_T(x)
    B = (1 + sum([x**k * eval_Tprime(x**k) for k in range(2, 50)]))
    C = sum([k * (x**k) * eval_Tprime(x**k) 
             + k * (x**(2*k)) * eval_Tsecond(x**k) for k in range(2, 50)])
    if pointed:
        D = sum([(k**2) * (x**k) * eval_Tprime(x**k) 
            + 3 * (k**2) * (x**(2*k)) * eval_Tsecond(x**k) 
            + (k**2) * (x**(3*k)) * eval_Tthird(x**k) for k in range(2, 50)])
        return C/(A**2) + 2 * (B**2) * (1-A)/(A**4) + D/B - ((C/B)**2)
    else:
        return C/A  + (1 - A) * (B**2)/(A**3)

def find_x_for_n(n, pointed):
    """
    Finds the Boltzmann parameter x in order to obtain
    a tree with expected size n.
    """
    return float(brentq(lambda x : expectation_size(x, pointed) - n, 0.2, rhoinv))

def compute_values(x):
    """
    Compute the values T(x**k) for k <= 50.
    """
    return [0] + [eval_T(x**k) for k in range(1, 51)]
  
