import subprocess
import numpy as np
import pandas as pd
from ete3 import Tree

def parse_phyml_file(outputfile, n_sites):
    sites = []
    with open(outputfile) as f:
        l = 0
        for line in f:
            if l >= 10:
                line = line.split()
                sites.append(line[-2])
            l+=1

    if len(sites) != n_sites:
        sites = np.repeat(np.nan, n_sites)
    return sites

def parse_phyml_stats(outputfile):
    l = np.nan
    with open(outputfile) as f:
        for line in f:
            if "Tree size:" in line:
                l = float(line.split("Tree size:")[1])
    return l

def parse_phyml_lik(outputfile):
    l = np.nan
    with open(outputfile) as f:
        for line in f:
            if "Log-likelihood:" in line:
                l = float(line.split("Log-likelihood:")[1])
    return l


def execute_phyml(path, cmd):
    if path is not None:
        phymlCommand = f"cd {path}; ./phyml {cmd} --quiet"
        _ = subprocess.run(phymlCommand, shell=True)
    else:
        phymlCommand = f"phyml {cmd} --quiet"
        _ = subprocess.run(phymlCommand, shell=True)


def run_phyml(filename, path_phyml, model, n_sites,
              ncat=4,
              output_file=None,
              tree_constraint=None,
              topology_constraint=None,
              remove_branch_length=False,
              return_likelihoods=False,
              run_free_rates=True):
    model = ['JC69', 'HKY85', 'GTR'][model]

    """
    -u tree_file -o r # optimize rates under the true tree 
    """
    cmd_g = f"-i {filename} -a e -c {ncat} -m {model} --print_site_lnl --run_id G"
    cmd_fr = f"-i {filename} --freerates -c {ncat} -m {model} --print_site_lnl --run_id FR"


    if tree_constraint is not None or topology_constraint is not None:
        if remove_branch_length:
            tree_constraint_tmp = tree_constraint + ".topology.tree"
            t = open(tree_constraint, "r").read().replace("[&R] ", "")
            tree = Tree(t, format=1)
            tree.write(outfile=tree_constraint_tmp, format=9)
            tree_constraint = tree_constraint_tmp
        if topology_constraint is not None:
            cmd_g = cmd_g + f" -u {tree_constraint} -o lr" #  optimizes rates and br lengths
            cmd_fr = cmd_fr + f" -u {tree_constraint} -o lr"
        else:
            cmd_g = cmd_g + f" -u {tree_constraint} -o r" # only optimizes rates
            cmd_fr = cmd_fr + f" -u {tree_constraint} -o r"


    # gamma model
    execute_phyml(path_phyml, cmd_g)
    sites_g = parse_phyml_file(filename + "_phyml_lk_G.txt", n_sites)
    tl_g = parse_phyml_stats(filename + "_phyml_stats_G.txt")

    # free rate model
    if run_free_rates:
        execute_phyml(path_phyml, cmd_fr)
        sites_fr = parse_phyml_file(filename + "_phyml_lk_FR.txt", n_sites)
        tl_fr = parse_phyml_stats(filename + "_phyml_stats_FR.txt")
    else:
        sites_fr = list(np.repeat(np.nan, len(sites_g)))
        tl_fr = tl_g * 0

    tbl = np.array([range(len(sites_g)), sites_g, sites_fr]).T
    p = pd.DataFrame(tbl)
    p.columns = ["site","rate_g","rate_fr"]

    if output_file is not None:
        p.to_csv(output_file)

    tbl = np.array([tl_g, tl_fr]).reshape((1,2))
    l = pd.DataFrame(tbl)
    l.columns = ["tl_g","tl_fr"]

    if return_likelihoods:
        lik_g = parse_phyml_lik(filename + "_phyml_stats_G.txt")
        if run_free_rates:
            lik_fr = parse_phyml_lik(filename + "_phyml_stats_FR.txt")
        else:
            lik_fr = lik_g * 0
        tbl = np.array([lik_g, lik_fr]).reshape((1, 2))
        lk = pd.DataFrame(tbl)
        lk.columns = ["lik_g", "lik_fr"]
        return p, l, lk
    else:
        return p, l