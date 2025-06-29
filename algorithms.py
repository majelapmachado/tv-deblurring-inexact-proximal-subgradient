# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:56:00 2025

@author: majela.penton@ufba.br
"""

import numpy as np
import time

from utilities_tv import (ApproxProx_fista, tlv)


def PepsilonSM(sigma, x0, x_opt, Ac, Ar, b, alpha, tau, maxIter, tol):
    start_time = time.time()
    fun_value = []
    err_psm = []
    x_pesm = x0.copy()
    
    
    err = 1
    k = 1
    coun_it_int = 0

    print(f"{'It':>3s}\t{'fun val':>8s}\t{'rel error':>10s}\t{'In It':>6s}")

    while k <= maxIter and err > tol:
        # Gradient step
        x_pesm_prev = x_pesm.copy()
        u = Ac.T @ (Ac @ x_pesm @ Ar.T - b) @ Ar
        y = x_pesm - alpha * u

        # Approximate prox
        x_pesm, int_it = ApproxProx_fista(alpha, y, sigma, tau, 2000)

        # Update function value and error
        fun_val = 0.5 * np.linalg.norm(Ac @ x_pesm @ Ar.T - b, 'fro')**2 + tau * tlv(x_pesm, 'iso')
        fun_value.append(fun_val)
        err = np.linalg.norm(x_pesm-x_pesm_prev,'fro') / np.linalg.norm(x_pesm,'fro')
        err_psm.append(err)
        coun_it_int += int_it

        if k % 10 == 0:
            print(f"{k:3d}\t{fun_val:8.6f}\t{err:10.6f}\t{int_it:6d}")
        k += 1

    tic_psm = time.time() - start_time
    return x_pesm, fun_value, tic_psm, err_psm, coun_it_int

