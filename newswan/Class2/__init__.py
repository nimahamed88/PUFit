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





class RunInfo :
    bx_l_r = 1e-3 * 1e6 / (2**18)
    progress_bar_des_l = 30
    allMarkerStyles = list(range(100,145)) + list(range(300,325)) + list(range(45)) + list(range(200,225)) + [236,336]
    colorscale = "Rainbow" #https://plotly.com/python/builtin-colorscales/
    
    @staticmethod
    def colorList() -> list:
        colors = []
        all___ = px.colors.qualitative.__dict__
        for a in all___:
            c = all___[a]
            if type(c) is list:
                colors += c 
        return colors

    
    @staticmethod
    def addnewrun(args):
        try:
            newrun = RunInfo(**args[1])
            newrun.parentRun = args[0]
            args[0]._subRuns.append(newrun)
        except Exception as e:
            print(e)
        args[0]._statusBar.update(1)

    @staticmethod
    def addnewrun_samelumibins(args):
        try:
            newrun = RunInfo(**args[1])
            newrun.parentRun = args[0]
            args[0]._subRunsSameLumiBins.append(newrun)
        except Exception as e:
            print('aa',e)
            raise e
        args[0]._statusBar.update(1)

        
    @staticmethod
    def setSim(args):
        try:
            args[0].setSimulation(args[1] , False)
        except Exception as e:
            print(e)
        args[2]._statusBar.update(1)
        
    @staticmethod
    def doFit(args):
        try:
            args[0].fit(False , args[1])
        except Exception as e:
            print(e)
        args[2]._statusBar.update(1)
        
    def __init__(self , run : int = 0, vname : str = "" , vbins : list = [] , lumi_var : str = "" , lumi_quantiles : list = [] , nbins_perq : int = 3,
                 lumi_hists = [], sub_runs : list = [] ,
                 max_pu : int = 100 , xsecs : np.ndarray = np.arange(60,100,0.1) , nthreads = 10 , _vals_ : np.ndarray = None):


        if len(lumi_quantiles) == 0:
            lumi_quantiles = [0]*( len(lumi_hists)+1 )
            if len(lumi_quantiles) == 1:
                raise ValueError("you need to at least speficy two quantiles for lumi/or pass lumi_hists")

        self._nthreads = nthreads
        self._vname = vname
        self._lumi_var = lumi_var
        self._run = run
        
           
        self._statusBar = tqdm(total=2*len(lumi_quantiles)+1+2*len(sub_runs)  , postfix="RUN {0}".format(self.run))
        self._subRuns = []
        if run == 0:
            argss = [(self , dict(run=r , vname=vname , vbins=vbins , lumi_var=lumi_var , 
                                                       lumi_quantiles=lumi_quantiles , nbins_perq=nbins_perq , lumi_hists=lumi_hists , sub_runs=[] , 
                                                       max_pu=max_pu , xsecs=xsecs )) for r in sub_runs]
            if nthreads > 0:
                with ThreadPoolExecutor(nthreads) as p:
                    p.map( RunInfo.addnewrun , argss )
            else:
                for a in argss:
                    RunInfo.addnewrun(a)
        
        self.xsecs = xsecs
        self.lumi_hists = []
        self.data_hists = []
        
        self._statusBar.set_description('loading'.ljust(RunInfo.progress_bar_des_l))
        if _vals_ is None:
            self._isSecondHand = False
        else:
            self._vals = _vals_
            self.nTotal = len( self._vals[self.lumi_var] )
            self._isSecondHand = True
            #print(self.vals)
        lumi_vals = self.vals[lumi_var]
        varVals = self.vals[vname]
        self._statusBar.update(1)

        
        self._statusBar.set_description('estimating quantiles'.ljust(RunInfo.progress_bar_des_l))
        if len(lumi_vals)==0:
            print(type(lumi_vals) , self.run)
            print(lumi_vals.shape)
            raise ValueError('file {0}'.format(self.filename))
        if sum(lumi_quantiles) == 0:
            lumi_limits = [ _ll.min for _ll in lumi_hists ] + [lumi_hists[-1].max]
        else:
            lumi_limits = np.quantile( lumi_vals , lumi_quantiles )
        self._statusBar.update(1)
     
        self._covmatrix = []
        for q_i in range(len(lumi_quantiles)-1):
            self._statusBar.set_description('q{0} pu dist'.format(q_i).ljust(RunInfo.progress_bar_des_l))

            h = np.histogram(lumi_vals , bins=nbins_perq , range=(lumi_limits[q_i],lumi_limits[q_i+1]) , density=True )

            lumiBin = lumiDistNumpy('run{0}q{1}'.format(run,q_i) , 
                          {(h[1][i],h[1][i+1]):float(h[0][i]) for i in range(nbins_perq)} ,
                          max_pu , self.xsecs , silent=True)
            self.lumi_hists.append(lumiBin)
            self._statusBar.update(1)

            self._statusBar.set_description('q{0}, {1} dist'.format( q_i , vname).ljust(RunInfo.progress_bar_des_l))
            seletedOnes = (lumi_vals < lumiBin.max) & (lumi_vals > lumiBin.min)
            h2 = np.histogram(varVals[seletedOnes] , bins=vbins)
            self.data_hists.append(discretepdf('h{0}_r{1}_q{2}'.format(vname,run,q_i),
                                               {(h2[1][i],h2[1][i+1]):float(h2[0][i]) for i in range(len(h2[0]))} ,
                                               unity=False ) )
            
            self._statusBar.set_description('q{0}, {1} cov'.format( q_i , vname).ljust(RunInfo.progress_bar_des_l))
            try:
                selMatrix = np.vstack([lumi_vals[seletedOnes] , varVals[seletedOnes] ])
                self._covmatrix.append( np.cov(selMatrix) )
            except Exception as e:
                print(e)
            self._statusBar.update(1)

        self._statusBar.set_description('lumi profile'.ljust(RunInfo.progress_bar_des_l))
        _ld_ = np.histogram(lumi_vals , bins=self.all_lumi_bins , density=True)
        self.lumi_distribution = discretepdf('run{0}_lumidist'.format(run) , 
                                             {(_ld_[1][i],_ld_[1][i+1]):float(_ld_[0][i]) for i in range(len(_ld_[0]))})
        try:
            self._covmatrix.append( np.cov(np.vstack([self.vals[self.lumi_var] , self.vals[self.vname]])))
        except Exception as e:
            print(run , e)
        self._statusBar.update(1)
        
        self._statusBar.set_description('SubRuns,FinalLumiProfile')
        self._subRunsSameLumiBins = []
        if run == 0:
            argss = [(self , dict(run=r.run , vname=vname , vbins=vbins , lumi_var=lumi_var , 
                                                       lumi_quantiles=[] , nbins_perq=nbins_perq , lumi_hists=self.lumi_hists , sub_runs=[] , 
                                                       max_pu=max_pu , xsecs=xsecs , _vals_=r.vals )) for r in self._subRuns]
            if nthreads > 0:
                with ThreadPoolExecutor(nthreads) as p:
                    p.map( RunInfo.addnewrun_samelumibins , argss )
            else:
                for a in argss:
                    RunInfo.addnewrun_samelumibins(a)
            
        
        self._statusBar.set_description('Run {0} Done'.format(self.run).ljust(RunInfo.progress_bar_des_l))
        self._statusBar.refresh()
        del self._statusBar
        if self.run == 0:
            del self.vals

    def __del__(self):
        for r in self._subRuns + self._subRunsSameLumiBins:
            del r
            
        del self.lumi_distribution
        for dh in self.data_hists:
            del dh
        for lh in self.lumi_hists:
            del lh
       
    @property
    def sigmaLumi(self) -> list:
        return [math.sqrt(self._covmatrix[i][0][0]) for i in range(self.nLumiBins+1) ]
    
    @property
    def sigmaVar(self) -> list :
        return [math.sqrt(self._covmatrix[i][1][1]) for i in range(self.nLumiBins+1) ]
    @property
    def correlation(self) -> list :
        return [self._covmatrix[i][0][1]/(self.sigmaLumi[i]*self.sigmaVar[i]) for i in range(self.nLumiBins+1) ]
    
    @property
    def nTotal(self) -> int :
        if hasattr(self , '_nTotalEvents'):
            return self._nTotalEvents
        elif len(self._subRuns) > 0:
            return sum([sr.nTotal for sr in self._subRuns])
        else:
            return -1
        
    @nTotal.setter
    def nTotal(self , v : int):
        self._nTotalEvents = v
    
    @property
    def parentRun(self) -> RunInfo:
        if hasattr(self , '_parentRun'):
            return self._parentRun
        else:
            return None
    @parentRun.setter
    def parentRun(self , p) :
        self._parentRun = p
        
    @property
    def lumiMinMax(self):
        if self.parentRun :
            return self.parentRun.lumiMinMax
        else:
            lumiBins = self.all_lumi_bins
            return [lumiBins[0] , lumiBins[-1]]
        
    @property
    def varMeanMinMax(self) :
        if self.parentRun:
            return self.parentRun.varMeanMinMax
        else:
            if not hasattr(self , "_varMeanMinMax"):
                allMeans = []
                for sr in self._subRuns + [self]:
                    for i in range(len(self.data_hists)):
                        _theMean = sr.data_hists[i].mean()
                        if not math.isnan(_theMean):
                            allMeans.append(_theMean)
                self._varMeanMinMax = (min(allMeans) , max(allMeans))
            return self._varMeanMinMax
        
    
    @property
    def runColor(self) -> str:
        return RunInfo.colorList()[self.idInList % len(RunInfo.colorList())] 
        
    @property
    def correlationColor(self) -> str:
        colors = []
        all___ = px.colors.qualitative.__dict__
        for a in all___:
            c = all___[a]
            if type(c) is list:
                colors += c 
        correlation = int( (1.0+self.correlation[-1])*100 )
        return colors[correlation % len(colors)]
        
    def plotMarkerStyle(self , colorLumiScale : int = 0) -> dict :
        ret = dict(symbol=RunInfo.allMarkerStyles[self.idInList % len(RunInfo.allMarkerStyles) ],
                    line_width=2,
                    size=10)
        
        ret["colorscale"] = RunInfo.colorscale
        if colorLumiScale == 1:
            ret["cmin"] = self.lumiMinMax[0]
            ret["cmax"] = self.lumiMinMax[1]
            ret['color'] = self.lumi_distribution.binRepresentatives
            ret['colorbar'] = dict(title="luminosity")
        elif colorLumiScale == 2:
            ret["cmin"] = self.varMeanMinMax[0]
            ret["cmax"] = self.varMeanMinMax[1]
            ret['colorbar'] = dict(title="average {0}".format(self.vname))
            ret['color'] = []
            for v,l in zip(self.data_hists , self.lumi_hists):
                _m = v.mean()
                if math.isnan(_m):
                    _m = ret["cmin"]-1
                ret['color'].extend([_m]*l.nbins)
            #if len(ret['color']) != self.nLumiBins:
            #print(ret["cmin"] , ret["cmax"] , ret['color'])
        elif colorLumiScale == 3:
            ret["cmin"]  , ret["cmax"] = 0,1
            _nt = self.nTotal
            ret['color'] = np.repeat( [v.integral()/_nt for v in self.data_hists] , [l.nbins for l in self.lumi_hists]+[0])
            ret['colorbar'] = dict(title="nEvents")
            #print(ret['color'])
        elif colorLumiScale == 4:
            ret["cmin"]  , ret["cmax"] = -1,1
            ret['color'] = np.repeat(self.correlation , [l.nbins for l in self.lumi_hists]+[0])
            ret['colorbar'] = dict(title="correlation")
        else:
            ret['color'] = self.runColor
        
        return ret
    
    @property
    def idInList(self) -> int:
        if self.parentRun :
            return 1+self.parentRun.subRunNumbers.index( self.run ) #+ 100*int(self._isSecondHand)
        else:
            return 0

    @property
    def subRunNumbers(self) -> list :
        return sorted([r.run for r in self._subRuns])
    
    @property
    def nthreads(self) -> int:
        return self._nthreads
    
    @property
    def lumi_var(self) -> str:
        return self._lumi_var
    @property
    def vname(self) -> str:
        return self._vname
    @property
    def vals(self):
        if not hasattr(self , '_vals'):
            if self.run == 0:
                self._vals = {a:np.array([]) for a in [self.lumi_var,self.vname] }
                for sr in self._subRuns:
                    for a in sr.vals.keys():
                        self._vals[a] = np.append( self._vals[a],sr.vals[a] )
                self.nTotal = len( self._vals[self.lumi_var] )
            else:
                with uproot.open(self.filename) as f:
                    tree = f['Events']
                    self._vals = tree.arrays([self.lumi_var,self.vname] , library='np')
                    self._vals[self.lumi_var] *= RunInfo.bx_l_r
                    self.nTotal = len( self._vals[self.lumi_var] )
        #print(self.nTotal, self.run)
        return self._vals
    
    @vals.deleter
    def vals(self):
        if hasattr(self , "_vals"):
            del self._vals
        for sr in self._subRuns + self._subRunsSameLumiBins:
            del sr.vals
            
    @property
    def run(self) -> int:
        return self._run
    
    @property
    def all_lumi_bins(self) -> list:
        all_lumi_bins = [0.0]
        for lumi_bin in self.lumi_hists:
            all_lumi_bins.pop()
            all_lumi_bins += lumi_bin.bins_array
        return sorted(all_lumi_bins)
    
    def plot_lumi_distribution(self , g=None , subRuns : int = 0 , colorLumiScale : int = 0 , density = True):
        
        linecolor = self.correlationColor if colorLumiScale in [4] else self.runColor
        g = self.lumi_distribution.plot(g=g, scatterOpts=dict(mode='markers',
                                             marker=self.plotMarkerStyle(colorLumiScale),
                                                             line=dict(color=linecolor)) , density=density )
        if subRuns:
            g.update_layout(legend_xanchor="left" , legend_orientation="h", showlegend=True)
            for sr in self._subRuns if subRuns > 0 else self._subRunsSameLumiBins:
                g = sr.plot_lumi_distribution(g, subRuns , colorLumiScale , density=density )
                
        return g
        
    def setSimulation(self, sim : SimulationVSPu , sub_runs : bool = True ) :
        njobs = 2*len(self._subRuns)+len(self.lumi_hists) if sub_runs else len(self.lumi_hists)
        colour = 'red' if sub_runs else 'yellow'
        if not hasattr(self , '_statusBar'):
            self._statusBar = tqdm(total=njobs  , postfix="RUN {0}".format(self.run) , colour=colour)
        if sub_runs:
            with ThreadPoolExecutor(self.nthreads) as p:
                p.map( RunInfo.setSim , [(r , sim , self) for r in self._subRuns+[self]+self._subRunsSameLumiBins ] )
        else:
            self.predictions = []
            for lumi_bin in self.lumi_hists:
                self.predictions.append(sim.predict(lumi_bin))
                self._statusBar.update(1)

    def plotPUDists(self , xsec:float , g=None) :
        for lh in self.lumi_hists:
            if g is None:
                g = lh.PUDist.plot(param=xsec)
            else:
                lh.PUDist.plot(param=xsec , g=g)
                
        return g
    
    def plotDataDist(self , g=None , zoom = True):
        if self.parentRun is None:
            g = make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)
        row = 1
        col = 1
        for dh in self.data_hists:
            theName = "Run {0}".format(self.run) if row*col == 1 else "Run {0}{1}".format(self.run, row*2+col-3)
            dh.plot(g=g , barmode=not self.parentRun is None , traceOpts={'row':row, 'col':col} , trimZeros=zoom,
                    scatterOpts={'marker_color':self.runColor , 'name':theName , 'legendgroup':"Run {0}".format(self.run) , 'showlegend':row*col==1})
            col += 1
            if col == 3:
                col = 1
                row += 1
        for sr in self._subRunsSameLumiBins:
            sr.plotDataDist(g)
        g.update_layout(barmode='stack')
        return g
    
    def plotPredictions(self , xsecs:list ):
        fig = make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)
        row = 1
        col = 1
        for pred in self.predictions:
            for xsec in xsecs:
                theName = "XSec={0}".format(xsec) if row*col == 1 else "XSec{0}{1}".format(xsec, row*2+col-3)
                pred.plot(param=xsec , g=fig , traceOpts={'row':row, 'col':col},
                         scatterOpts={'line_color':RunInfo.colorList()[int(xsec*10)%len(RunInfo.colorList())] , 'name':theName , 'legendgroup':"XSection={0}".format(xsec) , 'showlegend':row*col==1})
            col += 1
            if col == 3:
                col = 1
                row += 1
        return fig

    def plotRunPredictions(self , xsec:float , zoom : bool= True , fig = None):
        if self.parentRun is None:
            fig = make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)
        row = 1
        col = 1
        for pred in self.predictions:
            theName = "Run {0}".format(self.run) if row*col == 1 else "Run{0}{1}".format(self.run, row*2+col-3)
            pred.plot(param=xsec , g=fig , traceOpts={'row':row, 'col':col}, barmode=not self.parentRun is None , trimZeros=zoom,
                      scatterOpts={'marker_color':self.runColor , 'name':theName ,
                                   'legendgroup':"Run {0}".format(self.run) , 'showlegend':row*col==1}, norm=self.nTotal )
            col += 1
            if col == 3:
                col = 1
                row += 1
        for sr in self._subRunsSameLumiBins:
            sr.plotRunPredictions(xsec , zoom , fig)
        fig.update_layout(barmode='stack')
        return fig

    
    def fit(self , sub_runs = True , g = None ):
        if g is None:
            g = make_subplots(rows=self.nLumiBins+3 , cols = 3 ,
                              specs=[ [{'colspan':3}, None , None] , [{'colspan':3}, None , None] , [{'colspan':3}, None , None] ] + self.nLumiBins*[ [{},{},{}] ] )
            g.update_layout(height=(self.nLumiBins+3)*400, width=1200, title_text="chi2 values")
        if sub_runs:
            njobs = 2*len(self._subRuns)+1
            colour = 'red' if sub_runs else 'yellow'

            self._statusBar = tqdm(total=njobs  , postfix="RUN {0}".format(self.run) , colour=colour)
            with ThreadPoolExecutor(self.nthreads) as p:
                p.map( RunInfo.doFit , [(r , g , self) for r in self._subRuns+[self]+self._subRunsSameLumiBins ] )
            self._statusBar.refresh()
        else:
            self.fitResults = []
            finalres = {'x':[] , 'y':[] , 'ex':[] , 'ey':[]}
            theName_ = "Run {0}".format(self.run)

            run_fig_index = 1 + (1 if self._isSecondHand else 0) + (2 if self.parentRun is None else 0)
            for i in range(self.nLumiBins):
                theName = theName_.format( "" if i == 0 else i+1)
                pred = self.predictions[i]
                data = self.data_hists[i]
                chi2 = pred.chi2(data , self.xsecs)
                self.fitResults.append(chi2)
                chi2.plot(g=g , traceOpts={'row':i+1+3, 'col':run_fig_index} ,
                          scatterOpts={'marker_color':self.runColor , 'name':theName ,
                                       'legendgroup':"Run {0}".format(self.run) , 'showlegend':i==1 and run_fig_index>1})

                finalres['x'].append( (self.lumi_hists[i].max+self.lumi_hists[i].min)/2 )
                finalres['ex'].append( (self.lumi_hists[i].max-self.lumi_hists[i].min)/2 )

                finalres['y'].append( chi2.bestFit )
                finalres['ey'].append( chi2.bestFitError )

            additiona_trace_opt = {}
            if any([ey>10 for ey in finalres['ey'] ]):
                additiona_trace_opt['visible']='legendonly'
            g.add_trace(go.Scatter(x=finalres["x"], y=finalres["y"],
                        error_x=dict(type='data',array=finalres["ex"],visible=self.parentRun is None),
                        error_y=dict(type='data',array=finalres["ey"],visible=True), mode='markers',
                                      marker_color=self.runColor , name=theName_.format("BestFits"),
                                      legendgroup="Run {0}".format(self.run) , showlegend=False , **additiona_trace_opt), row=run_fig_index, col=1 )

        return g
    
    def postFitPlots(self):
        fig = make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)
        row = 1
        col = 1
        for i in range(self.nLumiBins):
            dh = self.data_hists[i]
            dh.plot(g=fig , traceOpts={'row':row, 'col':col} , 
                    scatterOpts=dict(mode='markers',marker=dict(color='black',size=6)))
            
            self.predictions[i].plot(param=self.fitResults[i].bestFit , g=fig ,
                                           traceOpts={'row':row, 'col':col} , norm=dh.integral() ,
                                           scatterOpts=dict(line=dict(color='red')))
            col += 1
            if col == 3:
                col = 1
                row += 1
        fig.update_layout(showlegend=False)
        return fig

    def pullPlots(self , maxtoshow=25 , smoothing=1.3):
        fig = go.Figure() #make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)
        
        row = 1
        col = 1
        for i in range(self.nLumiBins):
            dh = self.data_hists[i]
            norm=int( dh.integral() )
            x = []
            y = []
            for b in dh.binRepresentatives:
                x.append(b)
                d = float( dh.p(b) )
                pred = norm*float( self.predictions[i].p(param=self.fitResults[i].bestFit , val=b) )
                if pred == 0:
                    yval = 0
                else:
                    yval = (d-pred)/math.sqrt(pred)
                y.append(min(yval , maxtoshow) )
            fig.add_trace(go.Scatter(x=x , y=y , mode='lines' , line=dict(shape='spline', smoothing=smoothing)) ) #, row=row, col=col )
            col += 1
            if col == 3:
                col = 1
                row += 1
        #fig.update_layout(showlegend=False)
        return fig

    def NadjiehPullPlots(self , maxtoshow=25 , smoothing=1.3):
        fig = go.Figure() #make_subplots(rows=(self.nLumiBins//2) + (self.nLumiBins%2), cols=2)

        row = 1
        col = 1
        for i in range(self.nLumiBins):
            dh = self.data_hists[i]
            norm=int( dh.integral() )
            x = []
            y = []
            for b in dh.binRepresentatives:
                x.append(b)
                d = float( dh.p(b) )

                pdf_cent = float( self.predictions[i].p(param=self.fitResults[i].bestFit , val=b) )
                pdf_plus = float( self.predictions[i].p(param=self.fitResults[i].bestFit+self.fitResults[i].bestFitError , val=b) ) - pdf_cent
                pdf_minus = float( self.predictions[i].p(param=self.fitResults[i].bestFit-self.fitResults[i].bestFitError , val=b) )- pdf_cent
                pred = norm*pdf_cent
                pred_err = (abs(pdf_plus)+abs(pdf_minus))/2
                if pred == 0:
                    yval = 0
                else:
                    hamederr = math.sqrt(pred)
                    fEsq = pred * pdf_cent
                    sEsq = norm * norm * pred_err * pred_err
                    if fEsq+sEsq != 0:
                        yval = (d-pred)/math.sqrt(fEsq+sEsq) 
                    else:
                        yval = 0

                y.append(min(yval , maxtoshow) )
            fig.add_trace(go.Scatter(x=x , y=y , mode='lines' , line=dict(shape='spline', smoothing=smoothing)) ) #, row=row, col=col )
            col += 1
            if col == 3:
                col = 1
                row += 1
        #fig.update_layout(showlegend=False)
        return fig
 

    @property
    def nVarBins(self):
        return self.data_hists[0].nbins

    def chiSquared(self , lumibin):
        dh = self.data_hists[lumibin]
        norm=int( dh.integral() )
        ret = 0
        for b in dh.binRepresentatives:
            d = float( dh.p(b) )
            pred = norm*float( self.predictions[lumibin].p(param=self.fitResults[lumibin].bestFit , val=b) )
            ret += (d-pred)**2/pred
        return ret

    def aggregateFitRes2(self):
        _x_y = {}
        _x_yerr = {}
        _x_w = {}
        _x_corr = {}
        _x_chi2 = {}
        for sr in self._subRuns:

            for i in range(self.nLumiBins):
                lval = float(sr.lumi_hists[i].max+sr.lumi_hists[i].min)/2 
                #print(lval)
                chi2 = sr.fitResults[i]
                if chi2.atBorder:
                    continue
                y = chi2.bestFit
                y_err = chi2.bestFitError
                chi2value = 1 - stats.chi2.cdf( sr.chiSquared(i) , self.nVarBins )
                #if chi2value>10000:
                #    chi2value=10000
                if lval not in _x_y:
                    _x_y[lval] = [y]
                    _x_yerr[lval] = [y_err]
                    _x_w[lval] = [1.0/(y_err**2)]
                    _x_corr[lval] = [sr.correlation[i]]
                    _x_chi2[lval] = [chi2value]
                else:
                    _x_y[lval].append(y)
                    _x_yerr[lval].append( y_err )
                    _x_w[lval].append(1.0/(y_err**2))
                    _x_corr[lval].append(sr.correlation[i])
                    _x_chi2[lval].append(chi2value)

        _x = sorted(_x_y.keys())
        #print(_x)
        _y = []
        _yerr = []
        color = []
        for x in _x:
            if len(_x_y[x])>1:
                print(x)
            _y.append(np.average(np.array(_x_y[x]) , weights=np.array(_x_w[x])))
            #color.append(np.average(np.array(_x_corr[x]), weights=np.array(_x_w[x])))
            color.append(np.average(np.array(_x_chi2[x]), weights=np.array(_x_w[x])))
            _yerr.append(math.sqrt( np.average( (np.array(_x_y[x])-_y[-1])**2 , weights=np.array(_x_w[x]))) )
        g = go.Figure() 
        g.add_trace(go.Scatter(x=_x, y=_y,
                        error_y=dict(type='data',array=_yerr,visible=True), mode='markers',
                                      marker_color=color , marker_colorscale=RunInfo.colorscale , marker_colorbar = dict(title="correlation") ) )
        return g

    def aggregateFitRes(self):
        _x = []
        _y = []
        _yerr = []
        for i in range(self.nLumiBins):
            _x.append( (self.lumi_hists[i].max+self.lumi_hists[i].min)/2 )
            vs = []
            ws = []
            for sr in self._subRunsSameLumiBins:
                chi2 = sr.fitResults[i]
                vs.append( chi2.bestFit )
                ws.append( 1.0/(chi2.bestFitError**2) )
            vnp , wnp = np.array(vs) , np.array(ws)
            _y.append( np.average(vnp , weights=wnp) )
            _yerr.append( math.sqrt( np.average( (vnp-_y[-1])**2 , weights=wnp ) ) )
        g = go.Figure() 
        g.add_trace(go.Scatter(x=_x, y=_y,
                        error_y=dict(type='data',array=_yerr,visible=True), mode='lines+markers',
                                      marker_color=self.runColor) )
        return g
    
    @property
    def nLumiBins(self) -> int:
        return len(self.lumi_hists)
    @property
    def filename(self) -> str:
        return '/eos/user/c/cmstandi/PURunIIFiles/R{0}/all.root'.format(self.run)
    
    def aggregateFitRes4(self):
        g = make_subplots(self.nLumiBins, cols=1)
        g.update_layout(height=self.nLumiBins*400, width=1200, title_text="best xsection values per bin")
        for i in range(self.nLumiBins):
            lval = float(self.lumi_hists[i].max+self.lumi_hists[i].min)/2 

            _x_y = {}
            _x_yerr = {}
            _x_yerr2 = {}
            _x_w = {}
            _x_corr = {}
            _x_chi2 = {}
            _x_diffBestAvg = {}
            for x in range(self.nVarBins):
                theX = self.data_hists[0].binRepresentatives[x]
                _x_y[theX] = self.fitResults[i].Bins[x].bestFit
                all_ys = [a.fitResults[i].Bins[x].bestFit for a in self._subRunsSameLumiBins if not a.fitResults[i].Bins[x].atBorder] # and a._obs/a._nTotal > 0.000003]

                if len(all_ys)==0:
                    all_ys.append(_x_y[theX])
                avg = sum(all_ys)/len(all_ys)
                y_max = max(all_ys)
                y_min = min(all_ys)

                _x_yerr[theX] = y_max - _x_y[theX]
                _x_yerr2[theX] = _x_y[theX] - y_min
                if y_min >  _x_y[theX] or  y_max < _x_y[theX] :
                    #print('something wrong' , y_min , _x_y[x] , y_max)
                    ...
            _x = sorted(_x_y.keys())
            _y = []
            _yerr = []
            _yerr2 = []
            color = []
            for x in _x:
                _y.append(_x_y[x])
                color.append(0)
                _yerr.append(_x_yerr[x])
                _yerr2.append(_x_yerr2[x])

            g.add_trace(go.Scatter(x=_x, y=_y,
                            error_y=dict(type='data',array=_yerr,visible=True , symmetric=False,arrayminus=_yerr2), mode='lines' , line_color=RunInfo.colorList()[i]) , row=i+1, col=1  )
                                          #marker_color=color , marker_colorscale=RunInfo.colorscale , marker_colorbar = dict(title="correlation") ) )
            g.add_shape(type="line",
                        xref="x", yref="y",
                        x0=min(_x), y0=self.fitResults[i].bestFit, x1=max(_x), y1=self.fitResults[i].bestFit,
                        line=dict(color=RunInfo.colorList()[i+100],width=3)
                        , col=1 , row=i+1)

            g.add_shape(type="line",
                        xref="x", yref="y",
                        y0=min(self.xsecs), x0=self.data_hists[i].mean(), y1=max(self.xsecs), x1=self.data_hists[i].mean(),
                        line=dict(color=RunInfo.colorList()[i+100],width=3)
                        , col=1 , row=i+1)


        return g
    
    def aggregateFitRes3(self):
        _x_y = {}
        _x_yerr = {}
        _x_yerr2 = {}
        _x_w = {}
        _x_corr = {}
        _x_chi2 = {}
        _x_diffBestAvg = {}
        for sr in [self]: #._subRuns:

            for i in range(self.nLumiBins):
                lval = float(sr.lumi_hists[i].max+sr.lumi_hists[i].min)/2 
                #print(lval)
                chi2 = sr.fitResults[i]
                #if chi2.atBorder:
                #    continue
                y_best = chi2.bestFit
                all_ys = [a.bestFit for a in chi2.Bins if not a.atBorder and a._obs/a._nTotal > 0.000003]
                all_ys2 = {a._binId:a.bestFit for a in chi2.Bins if not a.atBorder and a._obs/a._nTotal > 0.003}
                #print(all_ys2)
                if len(all_ys)<2:
                    continue
                #
                y = sum(all_ys)/len(all_ys)
                y_err = max(all_ys)-y
                y_err2 = y - min(all_ys)

                diff = 100.0*(y-y_best)/y
                chi2value = 1 - stats.chi2.cdf( sr.chiSquared(i) , 25 )

                #if chi2value>10000:
                #    chi2value=10000
                if lval not in _x_y:
                    _x_y[lval] = [y]
                    _x_yerr[lval] = [y_err]
                    _x_yerr2[lval] = [y_err2]
                    _x_w[lval] = [1.0/(y_err**2)]
                    _x_corr[lval] = [sr.correlation[i]]
                    _x_chi2[lval] = [chi2value]
                    _x_diffBestAvg[lval] = [diff]
                else:
                    _x_y[lval].append(y)
                    _x_yerr[lval].append( y_err )
                    _x_yerr2[lval].append( y_err2 )
                    _x_w[lval].append(1.0/(y_err**2))
                    _x_corr[lval].append(sr.correlation[i])
                    _x_chi2[lval].append(chi2value)
                    _x_diffBestAvg[lval].append( diff )

        _x = sorted(_x_y.keys())
        #print(_x)
        _y = []
        _yerr = []
        _yerr2 = []
        color = []
        for x in _x:
            if len(_x_y[x])>1:
                print(x)
            _y.append(np.average(np.array(_x_y[x]) , weights=np.array(_x_w[x])))
            #color.append(np.average(np.array(_x_corr[x]), weights=np.array(_x_w[x])))
            #color.append(np.average(np.array(_x_chi2[x]), weights=np.array(_x_w[x])))
            color.append(np.average(np.array(_x_diffBestAvg[x]), weights=np.array(_x_w[x])))
            _yerr.append( math.sqrt( np.average( np.array(_x_yerr[x])**2 , weights=np.array(_x_w[x] ) ) ) )
            _yerr2.append( math.sqrt( np.average( np.array(_x_yerr2[x])**2 , weights=np.array(_x_w[x] ) ) ) )
        g = go.Figure() 
        g.add_trace(go.Scatter(x=_x, y=_y,
                        error_y=dict(type='data',array=_yerr,visible=True , symmetric=False,arrayminus=_yerr2), mode='markers',
                                      marker_color=color , marker_colorscale=RunInfo.colorscale , marker_colorbar = dict(title="correlation") ) )
        return g
