import numpy as np

# from utils import arg_min_scalar, arg_max_scalar, min_scalar, max_scalar
from utils import *

import scipy.optimize
from scipy.integrate import quad  # general numerical integration
from scipy.optimize import root_scalar
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import NonlinearConstraint, Bounds

from math import log

##############################
# Some convenience functions #
##############################

# Some basic coding bounds.
# * delta is relative distance
# * r is relative rate
# * rho is relative rate of correctable errors (list or unique)


def check_rate(rate):
    if rate < 0 or rate > 1:
        raise ValueError(f"Invalid rate ({rate})")
        
        
def check_delta(delta):
    if delta < 0 or delta > 0.5:
        raise ValueError(f"Invalid relative distance ({delta})")
        
        
def check_rho(rho):
    if rho < 0 or rho > 0.25:
        raise ValueError(f"Invalid fraction of correctable errors ({rho})")


# use this for computing jacobians with bounds.  taken from
# https://stackoverflow.com/questions/52208363/scipy-minimize-violates-given-bounds
def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """
    bounds: list of tuples (lower, upper)
    """
    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad
    return gradient
        

def GV_unique_rate_vs_delta(delta):
    """
    GV lower bound on the rate achievable by a uniquely decodable code with relative distance delta
    """
    check_delta(delta)
    return 1 - H(delta)


def GV_unique_delta_vs_rate(rate):
    """
    the same bound but inverted: a lower bound on the relative distance achievable for a given rate
    """
    check_rate(rate)
    return Hinv(1 - rate)


######################################
# Decodability of Concatenated Codes #
######################################


#  Guruswami-Rudra '08, Explicit Codes Achieving List Decoding Capacity
#  https://www.cs.cmu.edu/~venkatg/pubs/papers/FRS-full.pdf
#  
#  Theorem 5.2:
#  For all 0 < r, R < 1 and all epsilon > 0, there exist poly-time constructible family of 
#  binary linear codes of rate at least R * r which can be list-decoded in polynomial time 
#  up to a fraction (1 - R) * H^{-1}(1 - r) - epsilon of errors.
def concat_efficient_list_rho_vs_params(r_in, r_out):
    """
    Inputs:
        r_in: the rate of the (linear) inner code in Guruswami-Rudra (Explicit Codes
                Achieving List Decoding Capacity: Error-correction with Optimal Redundancy)
        r_out: the rate of the outer (Folded Reed-Solomon) code in the Guruswami-Rudra contruction
    Output:
        rho = such that a concatenation of a rate-r_out FRS code with random rate-r_in linear
        inner codes is efficiently rho'-list decodable with high probability (for all rho' < rho).
    """
    check_rate(r_in)
    check_rate(r_out)
    return (1 - r_out) * Hinv(1 - r_in)
    # print(f"result: {result}")


def concat_comb_distance_vs_params(r_in, r_out):
    """
    Inputs:
        r_in  = the rate of the (random linear) inner codes in Thommesen
        r_out = the rate of an outer code in Thommesen
    Outputs:
        delta such that for all rho' < rho, the resulting code has relative distance delta with
            all but exponentially small probability as long as the outer code is MDS.
    """
    check_rate(r_in)
    check_rate(r_out)
    def objective(theta):
        return theta * Hinv(1 - r_in * (1 - (1 - r_out) / theta))
    result = min_scalar(objective, method='Bounded', bounds=(1 - r_out, 1))
    return result


# Note this is the same as concat_comb_distance_vs_params, but included for completeness
def concat_comb_list_rho_vs_params(r_in, r_out):
    """
    Inputs:
        r_in =  the rate of the (random linear) inner codes in Guruswami-Rudra
                (Concatenated codes can achieve list decoding capacity) [GR08a]
        r_out = the rate of the outer (Folded Reed-Solomon) code in the Guruswami-Rudra contruction
    Output:
        rho such that for all rho' < rho, the resulting code is combinatorially rho'-list decodable
            with all but exponentially small probability as long as the outer code is sufficiently folded.
    """
    # This is the Fact 1.6 from (version 1) of our paper, which just said this bound is implicit in GR08a
    # specifically, need coefficients of exponents in equation (10) of GR08a to be negative
    def objective(theta):
        return theta * Hinv(1 - r_in * (1 - (1 - r_out) / theta))
    return min_scalar(objective, method='Bounded', bounds=(1 - r_out, 1))


#######################################################
# Best known prior efficiently decodable binary codes #
#######################################################


# See Eq. (4) of https://www.cs.cmu.edu/~venkatg/pubs/papers/lin-zyablov.pdf
def BZ_rate_vs_efficient_unique_rho(rho):
    """
    Inputs:
        rho: fraction of errors in (0, 0.25)
    Output:
        R such that for all R' < R, there exists a rate-R' code efficiently uniquely decodable
        in the presence of a fraction rho of errors.
    Notes:
        Integral may be numerically unstable for small values of rho.
    """
    def integrand(x):
        return 1 / Hinv(1 - x)
    (integral, precision) = quad(integrand, 0, 1 - H(2 * rho))
    return 1 - H(2 * rho) - 2 * rho * integral

# See Theorem 5.1 and subsequent discussion in https://www.cs.cmu.edu/~venkatg/pubs/papers/BZ-ld.pdf
# Note that BZ_rate_vs_efficient_list_rho(rho) = BZ_rate_vs_efficient_unique_rho(0.5 * rho)
def BZ_rate_vs_efficient_list_rho(rho):
    """
    Inputs:
        rho: fraction of errors in (0, 0.25)
    Output:
        R such that for all R' < R, there exists a rate-R' code efficiently list-decodable in the presence of
        a fraction rho of errors.
    Notes:
        Integral may be numerically unstable for small values of rho.
    """
    def integrand(x):
        return 1 / Hinv(1 - x)
    (integral, precision) = quad(integrand, 0, 1 - H(rho))
    return 1 - H(rho) - rho * integral


def TR_rho_vs_rate(R):
    """
    Input:
        R: real in (0, 1)
    Output:
        rho such that for all rho' < rho, there exists a rate-R code ensemble that is
        with high probability efficiently uniquely decodable against a fraction rho' of errors.
        
        rho is given by the Thommesen-Rudra bound.
    """
    # Optimize over choice of rates for concatenated code ensembles
    def objective(r_in):
        r_out = R / r_in
        return min(concat_efficient_list_rho_vs_params(r_in, r_out),
                   0.5 * concat_comb_distance_vs_params(r_in, r_out))
    
    return max_scalar(objective, method='Bounded', bounds=(R, 1))


def TR_rate_vs_rho(rho):
    return -inverse(lambda r: TR_rho_vs_rate(-r), -1, 0, num_iters=20)(rho)


def best_prior_rate_vs_rho(rho):
    return max(TR_rate_vs_rho(rho), BZ_rate_vs_efficient_unique_rho(rho))



#######################################
# Contextually Unique Decoding Bounds #
#######################################


def CUD_TR_rho_vs_rate_and_sparsity(R, s):
    """
    Inputs:
        R: apparent rate in (0,1)
        s: ``sparsity'' in (0,1)
    Outputs:
        rho such that for all rho' < rho, there exists a rate rho' ensemble
            of probabilistic codes that are w.h.p. contextually uniquely decodable
            against rho' fraction of errors, under the assumption that message space is
            as sparse as specified.
    """
    # Optimize over ``apparent rate'' (rate before we pad)
    def objective(r_in):
        r_out = R / r_in
        rho_e = concat_efficient_list_rho_vs_params(r_in, r_out)
        rho_c = concat_comb_list_rho_vs_params(r_in, r_out)

        # Because combinatorial (rho, 0)-list decodability implies
        # combinatorial (rho', H(rho') - H(rho))-list decodability.
        relaxed_rho_c = Hinv(min(1, H(rho_c) + R * (1 - s)))
        return min(rho_e, 0.5 * relaxed_rho_c)

    return max_scalar(objective, method='Bounded', bounds=(R, 1))


def CUD_BZ_apparent_rate_vs_rho_and_sparsity(rho, s):
    """
    Inputs:
        rho: fraction of errors
        s: sparsity
    Outputs:
        rate r such that for all r' < r, 
    """
    # need to be efficiently list decodable for at least rho errors, but could be more.
    # no point in going above 2 * rho errors though (because then sparsity is unnecessary)
    def objective(rho_relaxed):
        r_efficient = BZ_rate_vs_efficient_list_rho(rho_relaxed)
        
        # Because combinatorial (rho, 0)-list decodability implies
        # combinatorial (rho', H(rho') - H(rho))-list decodability.
        
        # find max r' < r_efficient such that r_efficient - s * r' > H(2 * rho) - H(rho_relaxed)
        # if no such r' exists, then efficient rho_relaxed-list decoding is not sufficient, so rate is 0
        r_apparent = (r_efficient - (H(2 * rho) - H(rho_relaxed))) / s
        if r_apparent > r_efficient:
            r_apparent = r_efficient
        if r_apparent < 0:
            return 0
        else:
            return r_apparent
    
    return max_scalar(objective, method='Bounded', bounds=(rho, 2 * rho))


def CUD_TR_rate_vs_rho_and_sparsity(rho, s):
    # negations are so that the function we are inverting is monotonically *increasing*
    return -inverse(lambda R : CUD_TR_rho_vs_rate_and_sparsity(-R, s), -1, 0, num_iters=15)(rho)


def CUD_best_rate_vs_rho_and_sparsity(rho, s):
    return max(CUD_TR_rate_vs_rho_and_sparsity(rho, s),
               CUD_BZ_apparent_rate_vs_rho_and_sparsity(rho, s))
    

######################################################
# Unique Decoding Bounds via the ``Padding'' Context #
######################################################


def best_padding_TR_rho_vs_rate(r_real):
    """
    Inputs:
        r_real: a real in (0, 1).
    Output:
        rho = fraction of correctable errors for a rate-r' concatenated code
    """
    # Optimize over ``apparent rate'' (rate before we pad)
    def objective1(r_apparent):
        # Optimize over ``inner rate''
        def objective2(r_in):
            r_out = r_apparent / r_in
            rho_e = concat_efficient_list_rho_vs_params(r_in, r_out)
            rho_c = concat_comb_list_rho_vs_params(r_in, r_out)
            
            # Because combinatorial (rho, 0)-list decodability implies
            # combinatorial (rho', H(rho') - H(rho))-list decodability.
            relaxed_rho_c = Hinv(min(1, H(rho_c) + r_apparent - r_real))
            return min(rho_e, 0.5 * relaxed_rho_c)
        
        return max_scalar(objective2, method='Bounded', bounds=(r_apparent, 1))
    
    return max_scalar(objective1, method='Bounded', bounds=(r_real, 1))


def best_padding_TR_rate_vs_rho(rho):
    # negations are so that the inverted function is monotonically increasing
    return -inverse(lambda r: best_padding_TR_rho_vs_rate(-r), -1, 0, num_iters=20)(rho)


def best_padding_BZ_rate_vs_rho(rho):
    # need to be efficiently list decodable for at least rho errors, but could be more.
    # no point in going above 2 rho errors though (because then padding is unnecessary)
    def objective(rho_relaxed):
        r_efficient = BZ_rate_vs_efficient_list_rho(rho_relaxed)
        
        # Because combinatorial (rho, 0)-list decodability implies
        # combinatorial (rho', H(rho') - H(rho))-list decodability.
        return max(0, r_efficient - (H(2 * rho) - H(rho_relaxed)))
    
    return max_scalar(objective, method='Bounded', bounds=(rho, 2 * rho))