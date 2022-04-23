from __future__ import annotations
import uproot
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import plotly
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from typing import overload
from abc import ABC, abstractmethod
import numpy as np
import sympy
from sympy.integrals.transforms import laplace_transform
import random
import math
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
from scipy.optimize import curve_fit
class parametricValue(ABC):
    def __init__(self , x , y):
        self._y = y
        self._x = x
    def __del__(self):
        del self._x
        del self._y
        
    @abstractmethod
    def _oneCall(self, v:float) -> float:
        ...
    def _multiCall(self , v:np.ndarray) -> np.ndarray:
        ret = []
        for f in v:
            ret.append(self(float(f)))
        return np.array(ret)
    
    @overload
    def __call__(self , v : float) -> float :
        ...
    @overload
    def __call__(self , v:np.ndarray) -> np.ndarray:
        ...
    def __call__(self , v):
        if type(v) in [float,int]:
            return self._oneCall(v)
        else:
            return self._multiCall(v)
        
    def __str__(self):
        return str(self._y)
    
    def __repr__(self):
        return str(self)
    
    def __rmul__(self , other) -> parametricValue:
        return self*other
        
    @abstractmethod
    def __mul__(self , other) -> parametricValue:
        raise ValueError('* operator for {0} and {1} types is not implemented'.format(type(self) , type(other)))
        
    @abstractmethod
    def __add__(self, other) -> parametricValue:
        raise ValueError('+ operator for {0} and {1} types is not implemented'.format(type(self) , type(other)))
    
    @abstractmethod
    def __sub__(self , other) -> parametricValue:
        raise ValueError('- operator for {0} and {1} types is not implemented'.format(type(self) , type(other)))
        
    @abstractmethod
    def __truediv__(self, other) -> parametricValue:
        raise ValueError('/ operator for {0} and {1} types is not implemented'.format(type(self) , type(other)))
        
    def __matmul__(self , other : discretepdf) -> parametricValue:
        ret = None
        for b,v in other.vals:
            myval = self(b.representative)
            
            newterm = myval*v
            if ret is None:
                ret = newterm
            else:
                ret += newterm
        return ret




class fixedValue(parametricValue):
    def __init__(self , v : float):
        super(fixedValue, self).__init__(None , v)
        
    def _oneCall(self , v) -> float :
        return self._y
    
    def __call__(self , v:float = 1.0) -> float :
        return super(fixedValue , self).__call__(v)
    
    def __mul__(self , other) -> fixedValue:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return fixedValue(self._y*other)
        else:
            super(fixedValue,self)*other

    def __add__(self , other) -> fixedValue:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return fixedValue(self._y+other)
        else:
            super(fixedValue,self)+other

    def __sub__(self , other) -> fixedValue:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return fixedValue(self._y-other)
        else:
            super(fixedValue,self)-other
            
    def __truediv__(self , other) -> fixedValue:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return fixedValue(self._y/other)
        else:
            super(fixedValue,self)/other


class parametricValueNumpy(parametricValue):
    def __init__(self , xarray : np.array , yarray : np.array):
        super(parametricValueNumpy , self).__init__(xarray , yarray)
        
    def _oneCall(self , v:float) -> float :
        idx = (np.abs(self._x - v)).argmin()
        return float( self._y[idx] )
    
    def _multiCall(self , v:np.ndarray) -> np.ndarray:
        if v is self._x:
            return self._y
        else:
            warnings.warn('numpy parametricValue is better to be called for the ful set of it is x range')
            return super(parametricValueNumpy,self)._multiCall(v)
        
    def __iadd__(self , other):
        if other._y.dtype not in[ np.dtype('float') ,  np.dtype('float32')]:
            #print(other._y.dtype)
            other._y = other._y.astype(float)
            #print(other._y.dtype)
        self._y += other._y
        return self
    
    def __mul__(self , other) -> parametricValueNumpy:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return parametricValueNumpy(self._x , other*self._y)
        else:
            super(fixedValue,self)*other

    def __add__(self , other) -> parametricValueNumpy:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return parametricValueNumpy(self._x , other+self._y)
        else:
            super(fixedValue,self)+other

    def __sub__(self , other) -> parametricValueNumpy:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return parametricValueNumpy(self._x , self._y-other)
        else:
            super(fixedValue,self)+other
            
    def __truediv__(self , other) -> parametricValueNumpy:
        if type(other) is fixedValue:
            other = other()
        if type(other) in [int , float]:
            return parametricValueNumpy(self._x , self._y/other)
        else:
            super(fixedValue,self)/other



class parametricValueSympy(parametricValue):
    def __init__(self, xsym : sympy.core.symbol.Symbol , yfunc : sympy.core.expr.Expr ):
        super(parametricValueSympy , self).__init__(xsym , yfunc)
        
    def _oneCall(self , v:float) -> float :
        val = self._y.subs(self._x , v).evalf()
        if type(val) is sympy.core.numbers.Float:
            return float(val)
        else:
            warnings.warn("sympy function did not reutrn float for param={0} : {1} - zero is returned".format(v,val))
            return 0
    
    def _multiCall(self , v:np.ndarray) -> np.ndarray:
        if type(self._y ) is sympy.core.numbers.Zero:
            return np.zeros(len(v))
        
        lmbd = sympy.lambdify( self._x , self._y)
        vals = lmbd(v)
        np.nan_to_num(vals , nan=np.finfo(np.float32).eps , copy=False)
        return vals
        
    def __iadd__(self , other):
        self._y = sympy.Add(self._y , other._y)
        return self
    
    def __mul__(self , other) -> parametricValueSympy:
        if type(other) is fixedValue:
            other = other()
        if type(other) is int:
            return parametricValueSympy(self._x , sympy.Integer(other)*self._y)
        elif type(other) is float:
            return parametricValueSympy(self._x , sympy.Float(other)*self._y)
        else:
            return parametricValueSympy(self._x , other*self._y)

    def __add__(self , other) -> parametricValueSympy:
        if type(other) is fixedValue:
            other = other()
        if type(other) is int:
            return parametricValueSympy(self._x , sympy.Integer(other)+self._y)
        elif type(other) is float:
            return parametricValueSympy(self._x , sympy.Float(other)+self._y)
        else:
            return parametricValueSympy(self._x , other+self._y)

    def __sub__(self , other) -> parametricValueSympy:
        if type(other) is fixedValue:
            other = other()
        if type(other) is int:
            return parametricValueSympy(self._x , self._y-sympy.Integer(other))
        elif type(other) is float:
            return parametricValueSympy(self._x , self._y-sympy.Float(other))
        else:
            return parametricValueSympy(self._x , self._y-other)
            
    def __truediv__(self , other) -> parametricValueSympy:
        if type(other) is fixedValue:
            other = other()
        if type(other) is int:
            return parametricValueSympy(self._x , self._y/sympy.Integer(other))
        elif type(other) is float:
            return parametricValueSympy(self._x , self._y/sympy.Float(other))
        else:
            return parametricValueSympy(self._x , self._y/other)

class binning:
    def __init__(self , min : float , max : float , representative : float = float('inf') ):
        self._min = min
        self._max = max
        self.hash = random.randint(0,10000)
        if representative != float('inf'):
            if self.fallsHere(representative):
                self._representative = representative
            else:
                raise ValueError('representative should fall in the range')
        else:
            self._representative = (min+max)/2

    def __eq__(self, other : binning) -> bool:
        return self.min == other.min and self.max == other.max
    
    def __lt__(self, other : binning) -> bool :
        return self.min < other.min
        
    def __hash__(self) -> int:
        return self.hash
    
    @property
    def min(self) -> float:
        return self._min
    
    @property
    def max(self) -> float:
        return self._max
    
    @property
    def representative(self) -> float:
        return self._representative
    
    @property
    def length(self) -> float:
        return self.max - self.min
    
    def fallsHere(self , val : float) -> bool:
        return val > self.min and val < self.max
        
    def overlaps(self, anotherbin : binning) -> bool:
        return self.fallsHere(anotherbin.min) or self.fallsHere(anotherbin.max)
    
    def __str__(self):
        return "({0},{1}:{2})".format(self.min, self.max , self.representative)



class discretepdf:
    def __init__(self , name : str , vals : dict , paramName : str = "" , valType : type = parametricValue , unity = True):
        self._paramName = paramName
        self._name = name
        self._bins = []
        self._probs = []
        self._valType = valType
        _vals = {}
        for b,v in vals.items():
            if type(v) in [float , int]:
                if v < 0:
                    raise ValueError('value of bin {0} is negative, it is not accepted'.format(b))
                v = fixedValue(float(v))

            if not issubclass( type(v) , valType):
                raise TypeError('value should be either float or inherit from parametricValue class. it is {0},{1}'.format(v,type(v)))
                
            self._probs.append(v)
            if type(b) is binning:
                self._bins.append(b)
            elif type(b) in [list , tuple] and len(b) in [2,3]:
                self._bins.append(binning(*b))
            else:
                raise ValueError('vals should contain either binning object or list/tuple')
                
        for b1 in self._bins:
            for b2 in self._bins:
                if b1.overlaps(b2):
                    raise ValueError('bin {0} and bin {1} have overlap'.format(b1 , b2))

        if unity:
            integral_for_zero = self.integral()
            if  integral_for_zero != 1.0:
                warnings.warn("integral of the pdf is not equal to one, it is {0}".format(integral_for_zero))
            
        sorted_pairs = sorted(self.vals)
        tuples = zip(*sorted_pairs)
        self._bins , self._probs = [ list(tuple) for tuple in  tuples]
        
        ii_hash = 0
        for b in self._bins:
            b.hash = ii_hash
            ii_hash += 1
           
    def fitPBinomial(self, param=0.0):
        N = max(self.binRepresentatives)
        def binomial(x , p ):
            if p==0 or p==1:
                return np.zeros(len(x))
            q = 1.0 - p
            n_m_x = N-x
            #print(p,q)
            lny = math.log(p)*x + math.log(q)*n_m_x
            for n in range(int(N)+1):
                if n!=0:
                    lny += math.log(n)
                lny -= np.log(np.clip(x-n , 1 , None) )
                lny -= np.log(np.clip(n_m_x - n , 1 , None) )
            return np.exp(lny)
        x = self.binRepresentatives
        y = np.array([a(float(param)) for a in self._probs])
        
        popt, pcov = curve_fit(binomial, x, y , bounds=([0.8],[0.999] ) )
        return popt[0]

    @property
    def valType(self) -> type :
        return self._valType
    
    @property
    def paramName(self) -> str:
        return self._paramName
    
    @property
    def vals(self):
        return zip(self._bins , self._probs)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def nbins(self) -> int:
        return len(self._bins)
        
    @property
    def bins_array(self) -> list:
        return [b.min for b in self._bins] + [self._bins[-1].max]
        
    def integral(self , param : float = 0) -> float:
        return sum([b.length*v(param) for b,v in self.vals])
    
    @property
    def min(self) -> float:
        return self._bins[0].min
    
    @property
    def max(self) -> float:
        return self._bins[-1].max
    
    def fallsHere(self , val : float) -> bool:
        return (val > self.min) and (val < self.max)
    
    def findBin(self , val:float) -> int :
        if self.fallsHere(val):
            return [b.fallsHere(val) for b in self._bins].index(True)
        else:
            return -1
        
    def p(self , val : float , param : float = 0) -> float:
        binId = self.findBin(val)
        if binId < 0:
            return 0
        else:
            return self._probs[binId](param)
     
    def mean(self , param : float = 0) -> float:
        a1 = 0
        a2 = 0
        for b,p in self.vals:
            P = p(param)
            a1 += b.representative*P*b.length
            a2 += P*b.length
        return a1/a2
    
    def stdDev(self, param: float = 0 ) -> float:
        mean = self.mean(param)
    
    @property
    def binRepresentatives(self) -> list:
        return [b.representative for b in self._bins]
    
    def allProbs(self , param : float = 0) -> np.ndarray:
        return np.array([v(param) for v in self._probs])
    
    def integrateParam(self, param_pdf : discretepdf) -> discretepdf :
        ret = {}
        for b,v in self.vals:
            ret[b] = v@param_pdf
        other_param_name = param_pdf.paramName
        return discretepdf('{0}_for_{1}'.format(self.name , other_param_name) , ret , other_param_name )
    
    def plot(self , label = None , param : float = 0 , g : plotly.basedatatypes.BaseFigure = None , 
            scatterOpts : dict = {} ,traceOpts : dict = {} , norm=1.0 , density = True, barmode=False,trimZeros=False) -> plotly.basedatatypes.BaseFigure:
        if g is None:
            g = go.Figure()
        if label is None:
            label = self.name
            if self.paramName != "":
                label += ', {0}={1}'.format(self.paramName , param)

        scatterOpts['y'] = norm*self.allProbs(param)
        if not density:
            scatterOpts['y'] = [ scatterOpts['y'][i]*self._bins[i].length  for i in range(self.nbins) ]

        scatterOpts['x'] = self.binRepresentatives
        if trimZeros:
            nonZeroIds = [l for l in range(len(scatterOpts['y'])) if scatterOpts['y'][l] != 0]
            if len(nonZeroIds)==0:
                return g
            _min , _max = min(nonZeroIds) , max(nonZeroIds)
            scatterOpts['x'] = scatterOpts['x'][_min:_max]
            scatterOpts['y'] = scatterOpts['y'][_min:_max]
            
        if 'mode' not in scatterOpts:
            if barmode:
                pass
            else:
                scatterOpts['mode'] = 'lines'
        if 'name' not in scatterOpts:
            scatterOpts['name'] = label
        if barmode:
            if 'mode' in scatterOpts:
                print( scatterOpts.pop('mode') )
            traceOpts['trace'] = go.Bar(**scatterOpts)
        else:
            traceOpts['trace'] = go.Scatter(**scatterOpts)
        
        g.add_trace(**traceOpts)
        return g
    
    def produceToy(self , ntotal : int , param : float) -> discretepdf :
        return discretepdf('{0}_toy_{1}(n:{2},{3}:{4})'.format(self.name , np.random.randint(100) , ntotal,self.paramName , param) ,
                           {b:np.random.poisson(ntotal*v(param)) for b,v in self.vals},
                           unity=False )
    
    def chi2(self , data : discretepdf , parVals : np.ndarray ) -> NLL :
        ntotal = int(data.integral())
        #chi2vals = np.zeros(len(parVals))
        bins = []
        bid = 0
        for b,d in data.vals:
            br = b.representative
            pdf_val = self._probs[self.findBin(br)]
            pred_i_vals = pdf_val(parVals)
            #np.nan_to_num(pred_i_vals , nan=np.finfo(np.float32).eps , copy=False)
            #pred_i_vals[pred_i_vals==0] = np.finfo(np.float32).eps
            
            #pred_i_vals_log = np.log(pred_i_vals)
            #chi2vals += pred_i_vals*ntotal - pred_i_vals_log*d()
            bins.append(NLL(data.name , bid , parVals , pred_i_vals , int(d()) , ntotal ))
            bid += 1
        #np.nan_to_num(chi2vals , nan=np.nanmax(chi2vals) , copy=False)
        #minimum = np.nanmin(chi2vals)
        #chi2vals -= minimum
        
        return NLL(data.name, -1 ,  parVals , binNLLs=bins )
    
    def __del__(self):
        for a,b in self.vals:
            del a
            del b


class NLL:
    def normalize(self):
        np.nan_to_num(self._y , nan=np.nanmax(self._y) , copy=False)
        minimum = np.nanmin(self._y)
        self._y -= minimum
        
    def __init__(self , name : str , binId : int , x : np.ndarray , pred_i : np.ndarray = None , obs_i : int = None , ntotal : int = None , binNLLs : list = []):
        self._name = name 
        self._binId = binId
        self._x = x
        
        self._pred = pred_i
        self._obs = obs_i
        self._nTotal = ntotal
        
        self.Bins = binNLLs        
        if binId < 0:
            if len(binNLLs)==0 :
                raise ValueError("to create a NLL for multiple bins, you should pass binId=-1 and set binNLLs arguments")
            for bnll in binNLLs:
                if bnll._x is not self._x:
                    raise ValueError("x of all bins should be the same")
            self._y = np.zeros(len(x))
            for bnll in binNLLs:
                self._y += bnll._y
        

        else:
            if pred_i is None or obs_i is None or ntotal is None:
                raise ValueError("to create NLL for one bin, pred_i , obs_i and ntotal should have values")
            np.nan_to_num(self._pred , nan=np.finfo(np.float32).eps , copy=False)
            self._pred[self._pred==0] = np.finfo(np.float32).eps
            
            pred_i_vals_log = np.log(self._pred)
            self._y = self._pred*ntotal - pred_i_vals_log*obs_i
        self.normalize()

        
      
    @property
    def name(self) -> str:
        if self._binId < 0:
            return self._name
        else:
            return "{0}_bin{1}".format( self._name , self._binId )
    
    @property
    def minIndex(self) -> int:
        return np.argmin(self._y)
    
    @property
    def bestFit(self) -> float :
        return float(self._x[self.minIndex])
    
    @property
    def atBorder(self) -> bool:
        return self.minIndex == 0 or self.minIndex == len(self._x)-1
    
    @property
    def bestFitError(self) -> float:
        if self.atBorder:
            return -1
        dxN = self._x[self.minIndex-1]-self._x[self.minIndex]
        dyN = self._y[self.minIndex-1]-self._y[self.minIndex]
        errN = abs(dxN/dyN)

        dxP = self._x[self.minIndex+1]-self._x[self.minIndex]
        dyP = self._y[self.minIndex+1]-self._y[self.minIndex]
        errP = abs(dxP/dyP)
        
        return float( max( (errP+errN)/2 , (abs(dxP)+abs(dxN))/2 ) )

        
    def plot(self , label = None , g : plotly.basedatatypes.BaseFigure = None,
             scatterOpts : dict = {} ,traceOpts : dict = {} ) -> plotly.basedatatypes.BaseFigure:
        if g is None:
            g = go.Figure()
        if label is None:
            label = self.name

        scatterOpts['x'] = self._x
        scatterOpts['y'] = self._y
        scatterOpts['mode'] = 'lines'
        if 'name' not in scatterOpts:
            scatterOpts['name'] = label
        
        traceOpts['trace'] = go.Scatter(**scatterOpts)
        g.add_trace(**traceOpts)
        return g
    
    def __str__(self) -> str:
        return "{0} best fit: {1} +- {2}".format(self.name , self.bestFit , self.bestFitError)

    def __repr__(self) ->str:
        return str(self)
    
    def __del__(self):
        if self._x is not None:
            del self._x
        del self._y
        for bll in self.Bins:
            del bll
        if self._pred is not None:
            del self._pred

class lumiDist(ABC,discretepdf):
    def __init__(self , name : str, vals : dict , 
                 min_pu : int , max_pu : int ,
                 xsec_min : float , xsec_max : float , nbins : int):
        super(lumiDist , self).__init__(name , vals , valType=fixedValue)
        self._minPU = min_pu
        self._maxPU = max_pu
        self._xsec_min = xsec_min
        self._xsec_max = xsec_max
        self._nbins = nbins
        
    @property
    def min_pu(self) -> int:
        return self._minPU
    @property
    def max_pu(self) -> int:
        return self._maxPU
    @property
    def min_xsec(self) -> float:
        return self._xsec_min
    @property
    def max_xsec(self) -> float:
        return self._xsec_max
    @property
    def nbins_xsec(self) -> int:
        return self._nbins

    @property    
    @abstractmethod
    def PUDist(self) -> discretepdf :
        ...

    @PUDist.deleter    
    @abstractmethod
    def PUDist(self) :
        ...
        
    def __del__(self):
        del self.PUDist
        super( type(self) , self).__del__()

class lumiDistSympyLaplace(lumiDist):
    def __init__(self , name : str , vals : dict , max_pu : int):
        super(lumiDistSympyLaplace , self).__init__(name , vals , 0 , max_pu , 0 , 1000 , -1)
        self.l = sympy.Symbol('lumi' , real=True, positive=True)
        
        array = [(0, self.l < self.min)]
        for l,v in self.vals:
            array.append( (v() , self.l<l.max ) )
        array.append((0 , True ) )
        self.sympyDist = sympy.Piecewise( *array )
        
        self.sigma = sympy.Symbol('sigma' , real=True, positive=True)
        self.cov_lumi_dist = laplace_transform(self.sympyDist , self.l , self.sigma)[0]
        self.pu_prob_for_sigma = {(-0.5,0.5):parametricValueSympy(self.sigma , self.cov_lumi_dist)}
        
        last_diff = self.cov_lumi_dist.diff(self.sigma)
        n_factoriel = 1
        negOne_n = -1
        sigma_n = self.sigma
        for n in range(1,self.max_pu+1):
            n_factoriel *= n
            fn = negOne_n * (sigma_n) * last_diff / n_factoriel
    
            negOne_n *= -1
            sigma_n *= self.sigma
            last_diff = sympy.simplify( last_diff.diff(self.sigma) )
    
            self.pu_prob_for_sigma[(n-0.5,n+0.5)] = parametricValueSympy(self.sigma , fn)
            print(n, sep=',', end=',', flush=True)
        self._PUDist = discretepdf( "puDist_{0}".format(self.name) , self.pu_prob_for_sigma , "sigma" , parametricValueSympy)
        

    @property
    def PUDist(self) -> discretepdf:
        return self._PUDist
    
    @PUDist.deleter
    def PUDist(self):
        del self._PUDist
    
class lumiDistNumpy(lumiDist):
    def __init__(self , name : str , vals : dict , max_pu : int , xsecs : np.ndarray ,silent=False):
        super(lumiDistNumpy , self).__init__(name , vals , 0 , max_pu , xsecs.min() , xsecs.max() , len(xsecs) )
        
        self.pu_prob_for_sigma = {}
        n_factoriel = 1
        for n in range(self.max_pu+1):
            if n != 0:
                n_factoriel *= n

            vals_1 = np.array([np.multiply( np.exp(-b.representative*xsecs),np.power(b.representative*xsecs,n))*v()*b.length/n_factoriel for b,v in self.vals])
            vals = np.sum(vals_1,0)
            #print(vals)
            self.pu_prob_for_sigma[(n-0.5,n+0.5)] = parametricValueNumpy(xsecs , vals)

            if not silent:
                print(n, sep=',', end=',', flush=True)
        self._PUDist = discretepdf( "puDist_{0}".format(self.name) , self.pu_prob_for_sigma , "sigma" , parametricValueNumpy)

    @property
    def PUDist(self) -> discretepdf:
        return self._PUDist

    @PUDist.deleter
    def PUDist(self):
        #del self._PUDist
        pass

    
class SimulationVSPu(discretepdf):
    
    @staticmethod
    def extract_vardist(args):
        try:
            nInts = args[1]
            pu = nInts
            args[0].pu_bins[nInts] = np.histogram( args[2][args[3]==nInts] , bins = args[0]._var_bins , density=True )[0]
            args[0]._statusBar.update(1)
        except Exception as e:
            print(e)
        
    def __init__(self , varname : str, year : int , fname : str = '' ,
                 var_bins : list = [] , var_min:float = -100 , var_max:float = -100 , var_nbins:int = -1 ,
                 pu_min : int = 0 , pu_max = 100 ,
                 mctune :int = 5 , apv : bool = False , nthreads = 10):
        self._fname = fname
        self._year = year
        self._varname = varname
        self._mctune = mctune
        self._apv = apv
        
        if len(var_bins)==0:
            if var_min >= var_max or var_nbins < 1:
                raise ValueError('you should specify variable range/binning explicitly')
            var_bins = [var_min+i*(var_max-var_min)/var_nbins for i in range(var_nbins+1)]
            
        self._var_bins = var_bins
        var_nbins = len(var_bins)-1
        self._statusBar = tqdm(total=pu_max+2  , postfix="SIMULATION")
        
        self.pu_bins = {}
        if pu_min == 0: #pu=0 is not simulated
            #pu_vals.append(0)
            self.pu_bins[0] = [1.0*(i==0) for i in range(var_nbins)]
            pu_min = 1
        with uproot.open(self.fileName) as file:
        #file = uproot.open(fSimulation)
            tree = file["PUAnalyzer"]["Trees"]["Events"]
            theArray = tree.arrays([self.varName , 'nInt'], 
                                   cut='(nInt>={0}) & (nInt<={1}) & ({2} >= {3}) & ({2} <= {4})'.format(
                                       pu_min , pu_max , self.varName , var_bins[0] , var_bins[-1]) ,
                                   library="np")
            
            allVarVals = theArray[self.varName]
            allPUS = theArray['nInt']
            self._statusBar.update(1)
            with ThreadPoolExecutor(nthreads) as p:
                p.map( SimulationVSPu.extract_vardist , [[self , i , allVarVals , allPUS] for i in range( pu_min , pu_max+1)] )
            del allVarVals
            del allPUS
            
        pu_vals_np = np.array(sorted([i for i in self.pu_bins.keys()]))
        pdfInputs = np.transpose( np.array([self.pu_bins[i] for i in pu_vals_np]) )
        super(SimulationVSPu , self).__init__( self.varName ,
                                              {(var_bins[i], var_bins[i+1]):parametricValueNumpy(pu_vals_np , pdfInputs[i] ) for i in range(var_nbins)} ,
                                              'pu' , parametricValueNumpy)
        self._statusBar.update(1)
        self._statusBar.refresh()
        del self._statusBar
        #del self.allVarVals
        #del self.allPUS
       
    def plotEfficiencies(self):
        pu_vals_np = np.array(sorted([i for i in self.pu_bins.keys()]))
        effs = np.array( [self.mean(float(pu))/pu for pu in pu_vals_np] )
        return discretepdf("recEff{0}_vs_pu".format(self.varName) , { (pu_vals_np[i],pu_vals_np[i+1]):fixedValue(effs[i]) for i in range(len(effs)-1) } , unity = False)
    
    @property
    def fileName(self) -> str :
        #print('/eos/user/c/cmstandi/PURunIIFiles/{0}/SingleNeutrino_CP{1}{2}.root'.format(self.year,self.mctune,"_APV" if self._apv else ""))
        return '/eos/user/c/cmstandi/PURunIIFiles/{0}/SingleNeutrino_CP{1}{2}.root'.format(self.year,self.mctune,"_APV" if self._apv else "")
    
    @property
    def mctune(self) -> int:
        return self._mctune
    
    @property
    def varName(self) -> str:
        return self._varname
    
    @property
    def year(self) -> int:
        return self._year
    
    def predict(self , pudist : lumiDist) -> discretepdf :
        return self.integrateParam(pudist.PUDist)
