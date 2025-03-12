import dendropy as dp
import random as rnd
import numpy as np
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.NewickIO import Writer
import dendropy as dp
from .utilities import print_update, biopython_msa_from_charmatrix

def simulateTree(ntips, mean_branch_length):
    t = dp.simulate.treesim.birth_death_tree(birth_rate=1.0, death_rate=0, num_total_tips=ntips)
    for edge in t.postorder_edge_iter():
        edge.length = rnd.expovariate(1 / mean_branch_length)

    return(t)

def getVCV(t):
    d = t.calc_node_root_distances(return_leaf_distances_only=False)
    taxa = t.taxon_namespace

    ntips = len(list(taxa))
    vcv = np.zeros((ntips,ntips))
    species = []

    for i in range(len(taxa)):
        species.append(taxa[i].label)
        for j in range(i, len(taxa)):
            node = t.mrca(taxa=[taxa[i],taxa[j]])
            vcv[i][j] = node.root_distance
            vcv[j][i] = node.root_distance

    return({"vcv": vcv, "species":species})

def pca(t):
    vcv = getVCV(t)
    C = vcv["vcv"]
    E, V = np.linalg.eigh(C)
    key=np.argsort(E)[::-1][:None]
    E, V = E[key], V[:, key]
    return({"eigenval": E, "eigenvect": V, "species":vcv["species"]})

def pca_from_ali(aln, verbose=False, tree_builder="nj"):
    """
    build UPGMA/NJ tree from biopython MSA and calc eigenvectors
    :param aln: dendropy char_matrix
    :return: dictionary -> {"eigenval": E, "eigenvect": V, "species":vcv["species"]}
    """
    # build UPGMA/NJ tree
    if tree_builder == "nj":
        constructor = DistanceTreeConstructor()
        tree_builder = constructor.nj
    else:
        constructor = DistanceTreeConstructor()
        tree_builder = constructor.upgma

    bio_msa = biopython_msa_from_charmatrix(aln)
    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(bio_msa)
    upgmatree = tree_builder(dm)
    ws = Writer([upgmatree])
    s = [i for i in ws.to_strings()][0]
    t = dp.Tree.get_from_string(s, "newick")
    if verbose:
        t.print_plot()
    # get eigenvectors
    return pca(t)


def simulateDNA(tree, seq_length,
                subs_model='GTR',
                freq=None,
                rates=None,
                ti_tv=0.5,
                scale=1.0,
                dir_shape_freq=10,
                dir_shape_rate=5,
                codon_pos_rates=None,
                seqgen_path=None):
    s = dp.interop.seqgen.SeqGen()
    if seqgen_path:
       s.seqgen_path = seqgen_path
    if subs_model == 'GTR':
        s.char_model = dp.interop.seqgen.SeqGen.GTR
    else:
        s.char_model = dp.interop.seqgen.SeqGen.HKY
    #-- OPTIONS
    # s.gamma_shape = 0.2
    # s.gamma_cats = 8
    # s.ti_tv = 0.5 # default = 0.5
    """
    A transition transversion ratio of 0.5 is the equivalent to equal rates of transitions and transversions 
    because there are twice as many possible transversions.
    """

    s.seq_len = seq_length
    s.scale_branch_lens = scale

    if subs_model != 'JC':
        if freq is not None:
            s.state_freqs = freq
        else:
            r = np.random.dirichlet([dir_shape_freq] * 4)
            s.state_freqs = list(r)
        if ti_tv == 0.5:
            ti_tv = np.random.uniform(2, 12)
            s.ti_tv = ti_tv

    if subs_model == 'GTR':
        if rates is not None:
            s.general_rates = rates
        else:
            r = np.random.dirichlet([dir_shape_rate] * 6)
            s.general_rates = list(r / r[-1])
            # r = [random.uniform(1, 3) for i in range(6)]
            # tot = sum(r)
            # s.general_rates = [i / tot for i in r]

    # print([i/s.general_rates[-1] for i in s.general_rates])

    if codon_pos_rates is not None:
        s.codon_pos_rates = codon_pos_rates

    return s.generate(tree)


def get_model_prm(n_blocks,
                  magnitude_frequency=5, # max magnitude
                  magnitude_rate=10,
                  gamma_shape=2
                  ):
    r = np.random.uniform(1, magnitude_frequency, (4, n_blocks))
    state_freqs = r / np.sum(r, axis=0)

    r = np.random.uniform(1, magnitude_rate, (6, n_blocks))
    general_rates = r / r[:,-1]

    if gamma_shape:
        scale = np.random.gamma(gamma_shape, 1 / gamma_shape, n_blocks)
    else:
        b = np.random.beta(0.7, 7, n_blocks)
        scale = b / np.mean(b)

def get_rnd_rates(n, rate_m="", p=None, verbose=False):
    model_list = ["Gamma", "Bimodal", "GBM", "Spike-and-slab", "Codon"]
    if p is None:
        p = np.ones(len(model_list)) / len(model_list)
    if rate_m == "autocorrelated":
        p[4] = 0 # codon model only when no sites blocks are used
        p /= np.sum(p)
    d = np.random.choice(model_list, p=p)
    if d == "Gamma":
        # gamma model
        r = np.exp(np.random.uniform(np.log(0.1), np.log(2))) # alpha prm
        scale = np.random.gamma(r, 1 / r, n)
    elif d == "Bimodal":
        # bimodal
        m = np.array([-1, 1])
        r = np.random.exponential(1)
        scale = np.exp(np.random.choice(m * r, n))
    elif d == "GBM":
        # Geometric Brownian (autocorrelated)
        r = np.random.uniform(0.02, 0.2) # rate
        s = np.random.normal(0, r, n+1)
        scale = np.exp(np.cumsum(s))
        if rate_m == "uncorrelated":
            scale = scale[np.random.choice(range(n), size=n, replace=False)]
    elif d == "Spike-and-slab":
        scale = np.exp(np.random.normal(0, 0.1, n))
        r = np.exp(np.random.uniform(np.log(0.01), np.log(0.1)))
        f = np.random.random(n)
        scale[f < r] = scale[f < r] * np.random.uniform(2,10)
    elif d == "Codon":
        # slow, very slow, high
        scale = np.exp(np.random.normal(0, 0.1, n)) # rate second position
        rnd_start = np.random.choice(range(3))
        indx_first_position = np.arange(rnd_start, n, 3)
        """
        1st -> 3rd
        0   -> 2
        1   -> 0
        2   -> 1
        """
        if rnd_start > 0:
            third = rnd_start - 1
        else:
            third = 2
        indx_third_position = np.arange(third, n, 3)
        scale[indx_first_position] *= np.random.uniform(1, 5)
        scale[indx_third_position] *= np.random.uniform(5, 15)
        r = indx_first_position

    if verbose:
        print(d, r)

    return scale / np.mean(scale), d, r

if __name__ == "__main__":
    for sim_i in range(1):
        # 1. Simulate a tree and get the eigenvectors
        print_update("simulating tree...")
        t = simulateTree(50, 0.5)  # args are: ntips, mean branch lengths
        x = pca(t)  # x is a dict with: "eigenval"-> eigenvalues; "eigenvect"-> eigenvectors; "species"-> order of labels
        n_sites = 1000
        blocks = 10
        n_sites_block = int(n_sites / blocks)

        # 2. Set the rates for each site and the number of sites of a given rate
        # vector specifying the number of sites under a specific rate (default: 1)
        sites_per_scale = list(np.ones(blocks) * n_sites_block)

        # 3. Set the model parameters (ie frequency and substitution rates)
        # scale is a vector holding the rates per site -> scaling factor to increase/decrease branch lengths
        scale = list(np.random.gamma(10, 1/10, blocks))

        # draw some random values for the frequency vector
        freq = None

        # same for the substitution rates
        rates = None

        # 4. Create a generator to simulate sites_per_scale[i] sites with rate scale[i]
        sim = (simulateDNA(t, f[0], scale=f[1]) for f in zip(sites_per_scale, scale))

        # 5. Call the generator once and then loop through it to simulate all the sites
        print_update("simulating data...")
        aln = next(sim)
        for d in sim:
            # aln.add_char_matrix(d.char_matrices[0])  # add a new dataset,
            aln.char_matrices[0].extend_matrix(d.char_matrices[0]) #if instead we want to happened all sites in a single alignment

        # access each simulated sites with
        print(aln.char_matrices[0].as_string())  # or "fasta"; order is the same as for the tree (from T1 to Tn)
        # import pandascharm as pc
        #
        # df = pc.from_charmatrix(aln.char_matrices[0])
        # print(df.shape)
        #
        # ali = pc.to_bioalignment(df) # alphabet='generic_dna'
        # print(ali._records)
        # print("running simulation n.:", sim_i)