import sys
import os
import copy
import numpy as np
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import scipy.stats
import scipy.ndimage

def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def unique_unsorted(a_tmp):
    a = copy.deepcopy(a_tmp)
    indx = np.sort(np.unique(a, return_index=True)[1])
    u = a_tmp[indx]
    return u

def load_pkl(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        print("Didn't work!")
        # import pickle5
        # with open(file_name, 'rb') as f:
        #     return pickle.load(f)

def save_pkl(obj, out_file):
    with open(out_file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


"""
functions from: https://github.com/jmenglund/pandas-charm
"""

def frame_as_categorical(frame, include_categories=None):
    """
    Return a pandas DataFrame with each column treated as a
    categorical with unordered categories. The same categories
    are applied to all columns.
    Parameters
    ----------
    frame : pandas.DataFrame
    include_categories : list (default: None)
        Categories to add unless they are already present
        in `frame`.
    """
    include_categories = include_categories if include_categories else []
    current_categories = pd.unique(frame.values.ravel())
    current_categories_notnull = (
        current_categories[pd.notnull(current_categories)])
    categories = set(current_categories_notnull).union(include_categories)
    categorical = frame.apply(lambda x: pd.Series(x.astype('category')))
    unified_categorical = categorical.apply(
        lambda x: x.cat.set_categories(new_categories=categories))
    return unified_categorical


def generate_random_alignment(num_sequences=5, seq_length=30, categorical=True):
    # Define nucleotide states
    nucleotides = ['A', 'C', 'G', 'T']

    # Generate a dictionary of random sequences
    alignment_data = {
        f"Seq{i + 1}": np.random.choice(nucleotides, seq_length).tolist()
        for i in range(num_sequences)
    }

    # Create DataFrame
    frame = pd.DataFrame(alignment_data).transpose()

    if categorical:
        # Convert to categorical data type
        for col in frame.columns:
            frame[col] = pd.Categorical(frame[col], categories=nucleotides)

    return frame

def df_from_charmatrix(charmatrix, categorical=True):
    """
    Convert a DendroPy CharacterMatrix to a pandas DataFrame.
    Parameters
    ----------
    charmatrix : dendropy.CharacterMatrix
    categorical : bool (default: True)
        If True, the result will be returned as a categorical frame.
    """
    frame = pd.DataFrame()
    for taxon, seq in charmatrix.items():
        s = pd.Series(
            seq.symbols_as_list(), name=taxon.label)
        frame = pd.concat([frame, s], axis=1)
    if categorical:
        state_alphabet = charmatrix.state_alphabets[0].symbols
        new_frame = frame_as_categorical(
            frame, include_categories=state_alphabet)
    else:
        new_frame = frame
    return new_frame.transpose()


def biopython_msa_from_charmatrix(charmatrix):
    s = charmatrix.as_string('phylip').split()
    l = []
    for i in range(2, len(s), 2):  # skip first 2 rows
        l.append(SeqRecord(Seq(s[i + 1]), id=s[i]))

    aln = MultipleSeqAlignment(l)
    return aln


def calc_confusion_matrix(y,lab):
    prediction = np.argmax(y, axis=1)
    y_actu = pd.Categorical(lab, categories=np.unique(lab))
    y_pred = pd.Categorical(prediction, categories=np.unique(lab))
    df_confusion = pd.crosstab(y_actu, y_pred, margins=False, rownames=['True'], colnames=['Predicted'],dropna=False)
    return df_confusion

def calc_accuracy(y,lab):
    if len(y.shape) == 3: # if the posterior softmax array is used, return array of accuracies
        acc = np.array([np.sum(i==lab)/len(i) for i in np.argmax(y,axis=2)])
    else:
        prediction = np.argmax(y, axis=1)
        acc = np.sum(prediction==lab)/len(prediction)
    return acc

def calc_label_accuracy(y,lab):
    prediction = np.argmax(y, axis=1)
    label_accs = []
    for label in np.unique(lab):
        cat_lab = lab[lab==label]
        cat_prediction = prediction[lab==label]
        acc = np.sum(cat_prediction==cat_lab)/len(cat_prediction)
        label_accs.append(acc)
    return np.array(label_accs)


def get_r2(x, y, return_mean=False, return_median=True, indx=None):
    if indx is None:
        R = range(len(x))
    else:
        R = indx
    if len(x.shape) == 1:
        if indx is not None:
            x = x[indx]
            y = y[indx]
        x = 0 + x.reshape((1, x.shape[0]))
        y = 0 + y.reshape((1, y.shape[0]))
        R = [0]

    r2s = []
    for i in R:
        x_i = x[i]
        y_i = y[i]
        try:
            _, _, r_value, _, _ = scipy.stats.linregress(x_i[np.isfinite(x_i)], y_i[np.isfinite(x_i)])
            r2s.append(r_value ** 2)
        except:
            pass

    if return_median:
        return np.nanmedian(r2s)
    elif return_mean:
        return np.nanmean(r2s)
    else:
        return np.array(r2s)

def get_mse(x, y, return_mean=True, return_median=False, indx=None):
    if indx is None:
        se = (x - y)**2
    else:
        se = (x[indx] - y[indx]) ** 2

    if return_median:
        return np.nanmedian(se)
    elif return_mean:
        return np.nanmean(se)
    else:
        return np.array(se)

def get_nrmse(x, y,
              return_mean=True,
              return_median=False,
              range_normalized=True,
              mean_normalized=False,
              indx=None):
    if indx is None:
        R = range(len(x))
    else:
        R = indx
    if len(x.shape) == 1:
        if indx is not None:
            x = x[indx]
            y = y[indx]
        x = 0 + x.reshape((1, x.shape[0]))
        y = 0 + y.reshape((1, y.shape[0]))
        R = [0]

    mse = []
    val_range = []
    val_mean = []
    for i in R:
        mse.append(np.nanmean((x[i] - y[i])**2))
        val_range.append( np.max(y[i]) - np.min(y[i]) )
        val_mean.append(np.mean(y[i]))

    mse = np.array(mse)
    rmse = np.sqrt(mse)
    r_nrmse = rmse / np.array(val_range)
    m_rmse = rmse / np.array(val_mean)

    if range_normalized:
        res = r_nrmse
    elif mean_normalized:
        res = m_rmse
    else:
        res = rmse

    if return_median:
        return np.nanmedian(res)
    elif return_mean:
        return np.nanmean(res)
    else:
        return res


def get_avg_r2(Ytrue, Ypred):
    r2 = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        r2.append(get_r2(x[x > 0], y[x > 0]))
    res = {'mean r2': np.nanmean(r2),   # change to nanmean? check if this works for sqs data.
           'min r2': np.nanmin(r2),
           'max r2': np.nanmax(r2),
           'std r2': np.nanstd(r2)}
    return res


def print_msa_compare(mle_fr, mle_g, dle, true, indx=None, digits=3):
    g = [round(get_mse(mle_g, true, indx=indx), digits), round(get_r2(mle_g, true, indx=indx), digits)]
    print("Gamma Rate MSE:", g[0], "R2:", g[1])
    fr = [round(get_mse(mle_fr, true, indx=indx), digits), round(get_r2(mle_fr, true, indx=indx), digits)]
    print("Free Rate MSE:", fr[0], "R2:", fr[1])
    dl = [round(get_mse(dle, true, indx=indx), digits), round(get_r2(dle, true, indx=indx), digits)]
    print("RNN Rate MSE:", dl[0], "R2:", dl[1])
    res = {
        'mse': [g[0], fr[0], dl[0]],
        'R2': [g[1], fr[1], dl[1]]
    }
    return res
    #
    #
    #
    #
    # print("Gamma Rate MSE:", round(get_mse(mle_g[np.isfinite(mle_g)], true[np.isfinite(mle_g)], indx=indx), digits),
    #       "R2:", round(get_r2(mle_g[np.isfinite(mle_g)], true[np.isfinite(mle_g)], indx=indx), digits))
    # print("Free Rate MSE:", round(get_mse(mle_fr[np.isfinite(mle_fr)], true[np.isfinite(mle_fr)], indx=indx), digits),
    #       "R2:", round(get_r2(mle_fr[np.isfinite(mle_fr)], true[np.isfinite(mle_fr)], indx=indx), digits))
    # print("RNN Rate MSE:", round(get_mse(dle, true, indx=indx), digits),
    #       "R2:", round(get_r2(dle, true, indx=indx), digits))


def get_mape(x, y, return_mean=True, return_median=False, indx=None):
    y[y == 0] = 1e-10
    if indx is None:
        se = np.abs(x - y) / y
    else:
        se = np.abs(x[indx] - y[indx]) / y[indx]

    if return_median:
        return np.nanmedian(se)
    elif return_mean:
        return np.nanmean(se)
    else:
        return np.array(se)
def print_mape_compare(mle_fr, mle_g, dle, true, indx=None, digits=3):
    g = [round(get_mape(mle_g, true, indx=indx), digits), round(get_r2(mle_g, true, indx=indx), digits)]
    print("Gamma Rate MAPE:", g[0], "R2:", g[1])
    fr = [round(get_mape(mle_fr, true, indx=indx), digits), round(get_r2(mle_fr, true, indx=indx), digits)]
    print("Free Rate MAPE:", fr[0], "R2:", fr[1])
    dl = [round(get_mape(dle, true, indx=indx), digits), round(get_r2(dle, true, indx=indx), digits)]
    print("RNN Rate MAPE:", dl[0], "R2:", dl[1])
    res = {
        'mape': [g[0], fr[0], dl[0]],
        'R2': [g[1], fr[1], dl[1]]
    }
    return res


def calcCI(data, level=0.95):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])

#################################################
######     RevBayes utility functions      ######
#################################################

def get_discretized_site_rates(site_rates, ncat=10, log_rates=True, test=False):
    if log_rates:
        counts, discrete_rates = np.histogram(np.log(site_rates), bins=ncat)
        discrete_rates = np.exp(discrete_rates)
    else:
        counts, discrete_rates = np.histogram(site_rates, bins=ncat)

    indx = np.digitize(site_rates, bins=discrete_rates[:-1]) - 1

    mean_rates = scipy.ndimage.mean(site_rates, index=indx, labels=indx)

    unique_rates = np.unique(mean_rates)
    indices = np.zeros(len(mean_rates)).astype(int)
    for i in range(len(unique_rates)):
        indices[mean_rates == unique_rates[i]] = i

    # TODO: ensure np.mean(unique_rates[indices]) == 1
    if test:

        plt.plot(mean_rates)
        plt.plot(site_rates)
        plt.show()
        """
        TEST
        """
        rs = get_rnd_gen(125)
        site_rates = np.array(list(rs.random(10)) + list(rs.uniform(2, 3, 10)))
        counts, discrete_rates = np.histogram(np.log(site_rates), bins=6)
        discrete_rates = np.exp(discrete_rates)
        indx = np.digitize(site_rates, bins=discrete_rates[:-1]) - 1

        mean_rates = scipy.ndimage.mean(site_rates, index=indx, labels=indx)
        unique_rates = np.unique(mean_rates)
        indices = np.zeros(len(mean_rates)).astype(int)
        for i in range(len(unique_rates)):
            indices[mean_rates == unique_rates[i]] = i

        print(unique_rates[indices] == mean_rates)

    return unique_rates, indices



def print_RevB_vec(name, v):
    new_v = []
    if len(v) == 0:
        vec = "%s <- v()" % (name)
    elif len(v) == 1:
        vec = "%s <- v(%s)" % (name, v[0])
    elif len(v) == 2:
        vec = "%s <- v(%s, %s)" % (name, v[0], v[1])
    else:
        for j in range(0, len(v)):
            value = v[j]
            try:
                if np.isnan(v[j]): value = "NA"
            except:
                print(v[j], v)
            new_v.append(value)

        vec = "%s <- v(%s, " % (name, new_v[0])
        for j in range(1, len(v) - 1): vec += "%s," % (new_v[j])
        vec += "%s)" % (new_v[j + 1])
    return vec


def get_phyloCTMC_model(partitioned=False, inv_model=None, discretize_site_rate=0):
    if partitioned is False:
        p = """
# the sequence evolution model
seq ~ dnPhyloCTMC(tree=psi, Q=Q, siteRates=sr, %s type="DNA")

# attach the data
seq.clamp(data)

        """ % inv_model
    else:
        if discretize_site_rate > 0:
            p = """
###############################################
# Create partitions (discretized)
###############################################
"""
            for i in range(discretize_site_rate):
                p += """
part_rate[%s] <- data                              
part_rate[%s].excludeAll()                
part_rate[%s].includeCharacter(part_%s_indx) 
dat[%s] ~ dnPhyloCTMC( tree=psi, Q=Q, nSites=part_rate[%s].nchar(), siteRates=v(sr[%s]), type="DNA")           
dat[%s].clamp(part_rate[%s])  
                """ % tuple([i + 1 for j in range(9)])
        else:
            p = """
###############################################
# Create partitions
###############################################
idx = 1

for (i in 1:int(num_char)) {
    morpho_bystate[i] <- data                              
    morpho_bystate[i].excludeAll()                
    morpho_bystate[i].includeCharacter(i)                            

    ######################
    # Substitution Model #
    ######################    

    m_morph[idx] ~ dnPhyloCTMC( tree=psi,
                                Q=Q,
                                nSites=1,
                                siteRates=v(sr[idx]),
                                type="DNA")           

    m_morph[idx].clamp(morpho_bystate[i])                 

    idx = idx + 1                                       
    # idx
}
        """
    return p


################ Revbayes Script

def get_revBayes_script(ali_name, res_name=None, out_name=None, sr=None,
                        gamma_model=False, inv_model=False,
                        prior_bl=10.0, discretize_site_rate=0, log_discretize=True):
    rate_block = ""
    n_rate_classes = 0
    if res_name is None:
        res_name = ali_name

    if out_name is None:
        out_name = ali_name

    if sr is None:
        partitioned = False

        if gamma_model:
            rate_block = """
        # among site rate variation, +Gamma4
        alpha ~ dnUniform( 0, 10 )
        sr := fnDiscretizeGamma( alpha, alpha, 4, false )
        moves.append( mvScale(alpha, weight=2.0) )

                """
            res_name = res_name + "_G"
            out_name = out_name + "_G"

        if inv_model:
            inv_block = """
            # the probability of a site being invariable, +I
            p_inv ~ dnBeta(1,1)
            moves.append( mvBetaProbability(p_inv, weight=2.0) )

                    """
            inv_model = "pInv=p_inv, "

            rate_block = rate_block + inv_block
        else:
            inv_model = ""


    else:
        partitioned = True
        if discretize_site_rate > 0:
            discrete_rates, rate_indx = get_discretized_site_rates(sr,
                                                              ncat=discretize_site_rate,
                                                              log_rates=log_discretize)
            n_rate_classes = len(discrete_rates) # some rate classes might be assigned to 0 sites
            print(discrete_rates, rate_indx, discretize_site_rate)
            rate_block = """
# among site rate variation (discrete)
%s
                        """ % print_RevB_vec("sr", discrete_rates)
            for d in range(len(discrete_rates)):
                ind = np.where(rate_indx == d)[0]
                print("rate_block", ind, d)
                part_name = d + 1
                if len(ind) > 0:
                    rate_block += """
%s                
                    """ % print_RevB_vec("part_%s_indx" % part_name, ind + 1)

            res_name = res_name + "_DL%sd" % discretize_site_rate
            out_name = out_name + "_DL%sd" % discretize_site_rate
        else:
            rate_block = """
# among site rate variation
%s
            """ % print_RevB_vec("sr", sr)
            res_name = res_name + "_DL"
            out_name = out_name + "_DL"



    phylo_model = get_phyloCTMC_model(partitioned=partitioned, inv_model=inv_model,
                                      discretize_site_rate=n_rate_classes)

    # script
    s = """

### Read in sequence data for the gene
data = readDiscreteCharacterData("%s")

# Get some useful variables from the data. We need these later on.
num_taxa <- data.ntaxa()
num_branches <- 2 * num_taxa - 3
taxa <- data.taxa()
num_char <- data.nchar()

moves    = VectorMoves()
monitors = VectorMonitors()


######################
# Substitution Model #
######################

# specify the stationary frequency parameters
pi_prior <- v(1,1,1,1) 
pi ~ dnDirichlet(pi_prior)
moves.append( mvBetaSimplex(pi, weight=2.0) )
moves.append( mvDirichletSimplex(pi, weight=1.0) )


# specify the exchangeability rate parameters
er_prior <- v(1,1,1,1,1,1)
er ~ dnDirichlet(er_prior)
moves.append( mvBetaSimplex(er, weight=3.0) )
moves.append( mvDirichletSimplex(er, weight=1.5) )


# create a deterministic variable for the rate matrix, GTR
Q := fnGTR(er,pi) 


#############################
# Among Site Rate Variation #
#############################

%s

##############
# Tree model #
##############

# Prior distribution on the tree topology
topology ~ dnUniformTopology(taxa)
moves.append( mvNNI(topology, weight=num_taxa/2.0) )
moves.append( mvSPR(topology, weight=num_taxa/10.0) )

# Branch length prior
for (i in 1:num_branches) {
    bl[i] ~ dnExponential(%s)
    moves.append( mvScale(bl[i]) )
}

TL := sum(bl)

psi := treeAssembly(topology, bl)




###################
# PhyloCTMC Model #
###################

%s


############
# Analysis #
############

mymodel = model(psi)

# add monitors
monitors.append( mnScreen(TL, printgen=100) )
monitors.append( mnFile(psi, filename="%s.trees", printgen=10) )
monitors.append( mnModel(filename="%s.log", printgen=10) )

# run the analysis
mymcmc = mcmc(mymodel, moves, monitors)
mymcmc.run(generations=10000)


# summarize output
treetrace = readTreeTrace("%s.trees", treetype="non-clock")
# and then get the MAP tree
map_tree = mapTree(treetrace,"%s_MAP.tre", ccp=TRUE)
mcc_tree = mccTree(treetrace,"%s_MCC.tre")
mcr_tree = consensusTree(trace=treetrace, cutoff=0, file="%s_CT.tre")
q()

    """ % (
        ali_name,
        rate_block,
        prior_bl,
        phylo_model,
        res_name,
        res_name,
        res_name,
        res_name, res_name, res_name
    )

    with open(out_name, 'w') as f:
        f.writelines(s)


