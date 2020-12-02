import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=0.8)

from codingbounds import *


def gen_figure_1(resolution=100):
    # Here we use results on the list decodability of multi-level 
    # concatenated codes to obtain improved bounds on the *unique*
    # decodability of such codes.  In particular, we use the fact
    # that it is possible to list-decode up to the Blokh-Zyablov bound
    # to show efficient unique decoding beyond half the Blokh-Zyablov bound
    # in the high-rate regime.
    rho_range = np.linspace(0, 0.25, resolution)
    rate_range = np.linspace(0, 1, resolution)
    
    GV_rates = np.frompyfunc(lambda rho : GV_unique_rate_vs_delta(2 * rho), 1, 1)(rho_range)
    plt.plot(rho_range, GV_rates, label="Gilbert-Varshamov (inefficient)", linestyle="--")

    BZ_rates = np.frompyfunc(BZ_rate_vs_efficient_unique_rho, 1, 1)(rho_range)
    plt.plot(rho_range, BZ_rates, label="Blokh-Zyablov")
    
    TR_rhos = np.frompyfunc(TR_rho_vs_rate, 1, 1)(rate_range)
    plt.plot(TR_rhos, rate_range, label="Thommesen-Rudra")
    
    # Find range of rates for which we improve over BZ and TR
    # eyeball estimate is that the crossover point is rho = 0.05
    crossover_rho = root_scalar(lambda rho : best_padding_BZ_rate_vs_rho(rho) - best_prior_rate_vs_rho(rho),
                            bracket=(0.03, 0.08)).root
    
    advantage_rho_range = np.linspace(0, crossover_rho, resolution)
    
                            
                            
    our_rates = np.frompyfunc(best_padding_BZ_rate_vs_rho, 1, 1)(advantage_rho_range)
    plt.plot(advantage_rho_range, our_rates, color="red", label="Our Result")

    plt.title("Comparison of Rate / Error Tolerance Tradeoffs")
    plt.xlabel("Correctable Fraction of Errors")
    plt.ylabel("Data Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_plot.pdf')