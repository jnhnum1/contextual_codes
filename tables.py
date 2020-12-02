from codingbounds import *

def gen_CUD_table():
    print(r"\begin{tabular}{|l||*{6}{c|}}\hline")
    rhos = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    sparsities = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    print(r"\backslashbox{Sparsity}{Errors} & " +
          " & ".join(["{:7.2f}".format(rho) for rho in rhos]) +
          r" \\ \hline\hline")
    for s in sparsities:
        print("{:4.2f}".format(s) +
              " & ".join(["{:7.3f}".format(CUD_best_rate_vs_rho_and_sparsity(rho, s)) for rho in rhos]) + 
              r" \\" )
    print(r"\hline \end{tabular}")