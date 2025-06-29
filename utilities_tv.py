# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:40:27 2025

@author: majela.penton@ufba.br
"""

import numpy as np
from scipy.sparse import spdiags, kron, vstack, eye
from scipy.linalg import hankel, toeplitz
from scipy.sparse.linalg import norm




from scipy.linalg import svd




def tlv(X, mode='iso'):
    m, n = X.shape
    P1, P2 = Ltrans(X)
    if mode == 'iso':
        D = np.zeros((m, n))
        D[:m-1, :] = P1 ** 2
        D[:, :n-1] += P2 ** 2
        return np.sum(np.sqrt(D))
    elif mode == 'l1':
        return np.sum(np.abs(P1)) + np.sum(np.abs(P2))
    else:
        raise ValueError("Invalid TV mode")

def Lforward(P1, P2):
    m2, n2 = P1.shape
    m1, n1 = P2.shape
    if n2 != n1 + 1 or m1 != m2 + 1:
        raise ValueError('dimensions are not consistent')
    m, n = m2 + 1, n2
    X = np.zeros((m, n))
    X[:m-1, :] = P1
    X[:, :n-1] += P2
    X[1:m, :] -= P1
    X[:, 1:n] -= P2
    return -X

def Ltrans(X):
    m, n = X.shape
    P1 = X[1:m, :] - X[:m-1, :]
    P2 = X[:, 1:n] - X[:, :n-1]
    return P1, P2

def ProjectionS(P1, P2, n, m):
    A = np.pad(P1, ((0, 1), (0, 0)))**2 + np.pad(P2, ((0, 0), (0, 1)))**2
    A = np.sqrt(np.maximum(A, 1))
    P1 = P1 / A[:m-1, :]
    P2 = P2 / A[:, :n-1]
    return P1, P2





def normGradient(linhas, colunas):
    # Create the sparse difference matrix Dy
    Dy = spdiags([-np.ones(colunas), np.ones(colunas)], [0, 1], colunas, colunas)
    Dy = Dy.tolil()
    Dy[colunas-1, :] = 0  # Set last row to 0
    Dy = Dy.tocsr()
    
    # Create the sparse difference matrix Dx
    Dx = spdiags([-np.ones(linhas), np.ones(linhas)], [0, 1], linhas, linhas)
    Dx = Dx.tolil()
    Dx[linhas-1, :] = 0  # Set last row to 0
    Dx = Dx.tocsr()
    
    # Kronecker products
    DX = kron(Dx, eye(colunas))
    DY = kron(eye(linhas), Dy)
    
    # Stack and compute Frobenius norm
    grad = vstack([DX, DY])
    normgrad = norm(grad, 'fro')  # Frobenius norm
    
    return normgrad


def padPSF(PSF, m, n=None):
    """
    PADPSF Pad a PSF array with zeros to make it bigger.

    Parameters:
        PSF (numpy.ndarray): Array containing the point spread function (PSF).
        m (int or tuple): Desired dimension of padded array. 
                          If a scalar is provided, both dimensions are set to m.
        n (int, optional): Desired number of columns. If only m is provided, n = m.
    
    Returns:
        numpy.ndarray: Padded m-by-n array.
    """
    
    # Set default parameters if only one dimension is provided (m is scalar)
    if n is None:
        if isinstance(m, int):
            n = m
        else:
            m, n = m  # m is a tuple with (m, n)
    
    # Pad the PSF with zeros
    P = np.zeros((m, n))  # Create a zero matrix of shape (m, n)
    P[:PSF.shape[0], :PSF.shape[1]] = PSF  # Place the original PSF in the top-left corner
    
    return P



def kronDecomp(P, center, BC='zero'):
    """
    KRON_DECOMP Kronecker product decomposition of a PSF array.

    Parameters:
        P (numpy.ndarray): Array containing the point spread function (PSF).
        center (tuple): Indices of the center of PSF, P.
        BC (str): Boundary condition, options: 'zero', 'reflexive', 'periodic'. Default is 'zero'.

    Returns:
        Ar, Ac (numpy.ndarray): Matrices in the Kronecker product decomposition.
    """
    
    # Check inputs
    if P is None or center is None:
        raise ValueError('P and center must be given.')
    
    # Find the two largest singular values and corresponding singular vectors of P
    U, S, Vt = svd(P, full_matrices=False)
    S = np.diag(S)
    
    if S[1, 1] / S[0, 0] > np.sqrt(np.finfo(float).eps):
        print("Warning: The PSF, P is not separable; using separable approximation.")
    
    # Ensure nonnegative components in the vectors
    minU = np.abs(np.min(U[:, 0]))
    maxU = np.max(np.abs(U[:, 0]))
    if minU == maxU:
        U = -U
        Vt = -Vt
    
    # Compute the rank-one vectors
    c = np.sqrt(S[0, 0]) * U[:, 0]
    r = np.sqrt(S[0, 0]) * Vt[0, :]
    
    # Build the matrices based on the boundary condition
    if BC == 'zero':
        Ar = build_toep(r, center[1])
        Ac = build_toep(c, center[0])
    elif BC == 'reflexive':
        Ar = build_toep(r, center[1]) + build_hank(r, center[1])
        Ac = build_toep(c, center[0]) + build_hank(c, center[0])
    elif BC == 'periodic':
        Ar = build_circ(r, center[1])
        Ac = build_circ(c, center[0])
    else:
        raise ValueError('Invalid boundary condition.')
    
    return Ar, Ac

def build_toep(c, k):
    """
    Build a banded Toeplitz matrix from a central column and an index.

    Parameters:
        c (numpy.ndarray): The column vector.
        k (int): The index of the central column.

    Returns:
        T (numpy.ndarray): The resulting Toeplitz matrix.
    """
    n = len(c)
    col = np.zeros(n)
    row = np.zeros(n)
    col[:n-k+1] = c[k-1:n]
    row[:k] = c[k-1::-1]
    T = toeplitz(col, row)
    return T

def build_circ(c, k):
    """
    Build a circulant matrix from a central column and an index.

    Parameters:
        c (numpy.ndarray): The column vector.
        k (int): The index of the central column.

    Returns:
        C (numpy.ndarray): The resulting circulant matrix.
    """
    n = len(c)
    col = np.concatenate([c[k-1:n], c[0:k-1]])
    row = np.concatenate([c[k-1::-1], c[n-1:k-1:-1]])
    C = toeplitz(col, row)
    return C

def build_hank(c, k):
    """
    Build a Hankel matrix for separable PSF and reflexive boundary conditions.

    Parameters:
        c (numpy.ndarray): The column vector.
        k (int): The index of the central column.

    Returns:
        H (numpy.ndarray): The resulting Hankel matrix.
    """
    n = len(c)
    col = np.zeros(n)
    col[:n-k] = c[k:n]
    row = np.zeros(n)
    row[n-k+1:] = c[0:k-1]
    H = hankel(col, row)
    return H


def ApproxProx_fista(alpha, y, sigma, tau, maxIt=10):
    m, n = y.shape
    u1 = np.zeros((m-1, n))
    u2 = np.zeros((m, n-1))
    v1_old = u1.copy()
    v2_old = u2.copy()
    
    t_old = 1
    k = 1
    normgrad = normGradient(m, n)
    gamma = 1 / normgrad**2
    inv_tau = 1 / tau
    
    Aux1, Aux2 = Ltrans(alpha * Lforward(u1, u2) - y)
    v1, v2 = ProjectionS(inv_tau * (u1 - (gamma/alpha) * Aux1), inv_tau * (u2 - (gamma/alpha) * Aux2), n, m)
    v1 = tau * v1
    v2 = tau * v2
    
    t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
    u1 = v1 + (t_old - 1) / t_new * (v1 - v1_old)
    u2 = v2 + (t_old - 1) / t_new * (v2 - v2_old)
    
    w = Lforward(v1, v2)
    eps_aux = y - alpha * w
    epsilon = np.abs(np.linalg.norm(eps_aux, 'fro')**2 - np.linalg.norm(y, 'fro')**2 + 2*alpha*tau*tlv(eps_aux, 'iso') + np.linalg.norm(alpha * w, 'fro')**2)
   
    while epsilon > sigma**2 * np.linalg.norm(alpha * w, 'fro')**2 and k <= maxIt:
        v1_old = v1.copy()
        v2_old = v2.copy()
        t_old = t_new
        
        Aux1, Aux2 = Ltrans(alpha * Lforward(u1, u2) - y)
        v1, v2 = ProjectionS(inv_tau * (u1 - (gamma/alpha) * Aux1), inv_tau * (u2 - (gamma/alpha) * Aux2), n, m)
        v1 = tau * v1
        v2 = tau * v2
        
        t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        u1 = v1 + (t_old - 1) / t_new * (v1 - v1_old)
        u2 = v2 + (t_old - 1) / t_new * (v2 - v2_old)
        
        w = Lforward(v1, v2)
        eps_aux = y - alpha * w
        epsilon = np.abs(np.linalg.norm(eps_aux, 'fro')**2 - np.linalg.norm(y, 'fro')**2 + 2*alpha*tau*tlv(eps_aux, 'iso') + np.linalg.norm(alpha * w, 'fro')**2)
        k += 1

    return y - alpha * w, k - 1


