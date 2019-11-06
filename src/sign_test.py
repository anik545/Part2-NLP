from scipy.stats import binom
from math import ceil


# Return p-value for B is better than
def sign_test(systemAcorrects, systemBcorrects):
    plus = 0
    minus = 0
    null = 0
    for a, b in zip(systemAcorrects, systemBcorrects):
        if a == b:
            null += 1
        elif a and not b:
            plus += 1
        elif b and not a:
            minus += 1

    k = ceil(null/2) + min(plus, minus)
    N = 2*ceil(null/2) + plus + minus

    B = binom(N, 0.5)

    sig = 2*B.cdf(k)

    return sig
