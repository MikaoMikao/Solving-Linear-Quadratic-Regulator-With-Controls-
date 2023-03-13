# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""


from Ex_1_1_matrices import (x, sigma, H, M, D, C, R, n, tgrid, batch_size)
from Ex_1_1_methods import solve_riccati_ode
from Ex_1_1_methods import value_function
from Ex_1_1_methods import control_function

# 1.1 Solving LQR using the Riccati ODE:
S = solve_riccati_ode(H, M, D, C, R, n, tgrid)

print(S[0])
print(S[-1])

v = value_function(tgrid, x, sigma, S, batch_size)
print(v[0])
print(v[-1])

a = control_function(x, D, M, S, batch_size)
print(a[0])
print(a[-1])

# 1.2 LQR MC checks