import numpy as np

def decompose_kernel(L):
    D, V = np.linalg.eigh(L)
    D = np.real(D)
    D[D < 0] = 0
    idx = np.argsort(D)
    D = D[idx]
    V = np.real(V[:, idx])
    return D, V

def esym_poly(k, lam):
  N = int(lam.size + 1)
  k = int(int(k) + 1)
  E = np.zeros((k, N))
  E[0, :] = np.ones((1, N))
  for l in range(1, k):
    for n in range(1, N):
      E[l, n] = E[l, n-1] + lam[n-1]*E[l-1, n-1]

  return E

def expected_cardinality(lam):
    return np.sum(lam/(1+lam))

def E_Y(lam):
    return np.sum(lam/(1+lam))
