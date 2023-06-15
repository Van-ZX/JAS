from ast import Lambda
from platform import python_branch
from tkinter.tix import IMAGE
import numpy as np
from scipy import ndimage, signal

def cost_hill(X):
    wetCost = 1e10
    hp = np.array([[-0.25, 0.5, -0.25], [0.5, -1, 0.5], [-0.25, 0.5, -0.25]], dtype=np.float32)
    r_1 = signal.convolve2d(X.astype(np.float32), hp, boundary='symm', mode='same')
    lp_1 = np.ones([3, 3], dtype=np.float32)/9
    r_2 = signal.convolve2d(np.abs(r_1), lp_1, boundary='symm', mode='same')
    rho = 1/(r_2+1e-10)
    lp_2 = np.ones([15, 15], dtype=np.float32)/225
    rho = signal.convolve2d(rho, lp_2, boundary='symm', mode='same')
    rho[rho > 50] = wetCost
    rho[np.isnan(rho)] = wetCost
    rho_p = rho.copy()
    rho_m = rho.copy()
    rho_p[X == 255] = wetCost
    rho_m[X == 0] = wetCost
    return rho_p, rho_m


def getHt(pP1, pM1):
    # import ipdb
    # ipdb.set_trace()
    p = np.zeros([3, pP1.shape[0], pP1.shape[1]], dtype=np.float)
    I = np.ones([pP1.shape[0], pP1.shape[1]], dtype=float)
    p0 = I - pP1 - pM1
    p[0, :] = p0
    p[1, :] = pP1
    p[2, :] = pM1
    p[p == 0] = 1e-10

    H = - (p * np.log2(p))
    # H((P < np.spacing(1)) | (P > 1 - np.spacing(1))) = 0
    H[(p < 2.2204e-16) | (p > 1 - 2.2204e-16)] = 0

    Ht = H.sum()

    return Ht

def calc_lambda(rhop, rhom, message_length, n):
    l3 = 1e+3
    m3 = message_length + 1
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = np.zeros(rhop.shape, dtype=np.float)
        pM1 = np.zeros(rhom.shape, dtype=np.float)
        pP1 = np.exp(-l3 * rhop) / (1 + np.exp(-l3 * rhop) + np.exp(-l3 * rhom))
        pM1 = np.exp(-l3 * rhom) / (1 + np.exp(-l3 * rhop) + np.exp(-l3 * rhom))
        m3 = getHt(pP1, pM1)
        iterations = iterations + 1
        if iterations > 30:
            Lambda = l3
            return Lambda

    l1 = 0
    Lambda = 0
    lastm = 0

    while (float(abs(lastm/n - message_length/n)) > 1/1000.0) and (iterations < 300):
        Lambda = l1 + (l3 - l1) / 2.0
        pP1 = np.zeros(rhop.shape, dtype=np.float)
        pM1 = np.zeros(rhom.shape, dtype=np.float)
        pP1 = np.exp(-Lambda * rhop) / (1 + np.exp(-Lambda * rhop) + np.exp(-Lambda * rhom))
        pM1 = np.exp(-Lambda * rhom) / (1 + np.exp(-Lambda * rhop) + np.exp(-Lambda * rhom))
        m2 = getHt(pP1, pM1)
        if m2 < message_length:
            l3 = Lambda
            m3 = m2
            lastm = m3
        else:
            l1 = Lambda
            lastm = m2
        iterations = iterations + 1
    return Lambda

def hill_emb(X, rhoP1, rhoM1, m):
    n = X.size
    Lambda = calc_lambda(rhoP1, rhoM1, m, n)
    pChangeP1 = np.zeros(rhoP1.shape, dtype=np.float)
    pChangeM1 = np.zeros(rhoM1.shape, dtype=np.float)
    pChangeP1 = np.exp(-Lambda * rhoP1) / (1 + np.exp(-Lambda * rhoP1) + np.exp(-Lambda * rhoM1))
    pChangeM1 = np.exp(-Lambda * rhoM1) / (1 + np.exp(-Lambda * rhoP1) + np.exp(-Lambda * rhoM1))
    randChange = np.random.rand(X.shape[0], X.shape[1])
    Y = X.copy()
    Y[randChange < pChangeP1] = Y[randChange < pChangeP1] + 1
    Y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = Y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1
    return Y