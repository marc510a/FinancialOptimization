import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sympy as sp 
import cvxpy as cp

def f_transaction(lamb, x0, x1, z_bin, Q, returns, c0, c1):
    x = x0 + x1
    sigma = cp.quad_form(x, Q)
    # Return on total holdings minus transaction costs
    ret = returns @ x - c0*cp.sum(z_bin) - c1*cp.sum(x1)
    return lamb * sigma - (1 - lamb) * ret

def f_revision(lamb, x, yplus, yminus, zplus, zminus, Q, returns, c0, c1):
    sigma = cp.quad_form(x, Q)
    ret = returns @ x - c0*cp.sum(zplus) - c0*cp.sum(zminus) - c1*cp.sum(yplus) - c1*cp.sum(yminus)
    return lamb * sigma - (1 - lamb) * ret

def meanvariance_frontier(lamb, Q, returns, c0, c1, n, exist=None):
    if exist is None:
        x0=cp.Variable(n) #Weights at flat fee
        x1=cp.Variable(n) #Weights at linear fee
        z_bin=cp.Variable(n, boolean=True) #Binary variable for flat fee
        objective = cp.Minimize(f_transaction(lamb, x0, x1, z_bin, Q, returns, c0, c1))
        constraints=[cp.sum(x0+x1)==1, x0>=0, x0<=x_bar*z_bin, x1>=0, x1<=z_bin, cp.sum(z_bin)>=5, x0+x1>=0.01*z_bin]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        return (x0.value + x1.value), z_bin.value
    else:
        zplus=cp.Variable(n, boolean=True)
        zminus=cp.Variable(n, boolean=True)
        z_final = cp.Variable(n, boolean=True)
        x0=exist
        yplus=cp.Variable(n)
        yminus=cp.Variable(n)
        x=x0+yplus-yminus
        objective=cp.Minimize(f_revision(lamb, x, yplus, yminus, zplus, zminus, Q, returns, c0, c1))
        constraints = [cp.sum(x)==1, yplus>=0, yplus<=zplus, yminus>=0, yminus<=zminus, 
                   x0+yplus>=x_low_buy*zplus, x0+yplus<=x_high_buy*zplus+(1-zplus),
                   x0-yminus>=x_low_sell*zminus, x0-yminus<=x_high_sell*zminus+(1-zminus),
                   zplus + zminus <= 1, x >= 0, x>=0.01*z_final, cp.sum(z_final)>=5, x<=z_final]
        prob=cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        return (x0 + yplus.value - yminus.value), zplus.value, zminus.value