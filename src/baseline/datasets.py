import numpy as np
import re
#from tqdm import tqdm
from tqdm.notebook import tqdm

def load_1000genomes_demographics(path='../data/1000genomes/integrated_call_samples_v3.20130502.ALL.panel'):
    """
    sample	pop	super_pop	gender
    HG00096	GBR	EUR	male
    HG00097	GBR	EUR	female
    """
    dataset = []
    with open(path) as file:
        for line in file:
            dataset.append(line.strip('\n').split('\t'))
    dict = {}
    for subject in dataset[1:]:
        dict[subject[0]] = subject[1:]
    return dict

def load_1000genomes_large(path='../data/1000genomes/ALL.chr1.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf', n = 100000):
    """
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">
    ##contig=<ID=chr1>
    ##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated allele frequency in the range (0,1)">
    ##INFO=<ID=AC,Number=A,Type=Integer,Description="Total number of alternate alleles in called genotypes">
    ##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of samples with data">
    ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
    ##INFO=<ID=EAS_AF,Number=A,Type=Float,Description="Allele frequency in the EAS populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=EUR_AF,Number=A,Type=Float,Description="Allele frequency in the EUR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=AFR_AF,Number=A,Type=Float,Description="Allele frequency in the AFR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=AMR_AF,Number=A,Type=Float,Description="Allele frequency in the AMR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=SAS_AF,Number=A,Type=Float,Description="Allele frequency in the SAS populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=VT,Number=.,Type=String,Description="indicates what type of variant the line represents">
    ##INFO=<ID=EX_TARGET,Number=0,Type=Flag,Description="indicates whether a variant is within the exon pull down target boundaries">
    ##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096	HG00097	HG00099	HG00100	HG00101	HG00102	HG00103	HG00104	HG00105	HG00106	HG00107	HG00108	HG00109	HG00110	HG00111	HG00112	HG00113	HG00114	HG00115	...
    chr1	16103	.	T	G	.	PASS	AC=126;AN=5096;DP=29994;AF=0.02;EAS_AF=0;EUR_AF=0.04;AFR_AF=0.03;AMR_AF=0.03;SAS_AF=0.02;VT=SNP;NS=2548	GT	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	0|0	...
    chr1	51479	.	T	A	.	PASS	AC=536;AN=5096;DP=17461;AF=0.11;EAS_AF=0;EUR_AF=0.2;AFR_AF=0.02;AMR_AF=0.12;SAS_AF=0.22;VT=SNP;NS=2548	GT	1|0	0|0	0|0	0|0	0|0	0|0	0|1	0|1	1|0	0|0	0|1	0|1	0|0	1|0	0|0	1|0	1|0 ...
    """
    assert n % 2 == 0, 'n={} needs to be divisible by 2'.format(n)
    with open(path) as file:
        line = next(file)
        while line.startswith('##'): # skip info fields
            line = next(file)
        header = re.split('[\t|]+', line)[9:]
        dataset = np.zeros((n,len(header)))
        i = 0
        with tqdm(total=n) as pbar:
            while i < n:
                line = next(file)
                line = line.strip('\n')
                line = re.split('[\t|]+', line)[9:]
                line = np.array(line).astype(np.float32)
                dataset[i], dataset[i+1] = line[np.arange(0, len(line), 2)], line[np.arange(1, len(line), 2)] # left and right halves of pair
                i += 2
                pbar.update(2)
    return dataset, header

def represent_strarr_as_indices(arr):
    """
    arr: (m, n) array of strings
    returns (m, n) array of indices instead of strings, and an array mapping index -> string
    """
    unique_strings, index_array = np.unique(arr.flatten(), return_inverse = True)
    index_array = index_array.reshape(arr.shape)
    return index_array, unique_strings

def represent_genome_as_rank(base_pair_array):
    """
    base_pair_array: (num_people, genome_length) array of base pairs
    returns (num_people, genome_length) array of indices (0, 1, 2, or 3) representing the frequency rank of that base pair in the dataset (0 being most common). In all cases there is only 1 possible mutation so the array is binary.
    """
    base_pair_index_array, base_pairs = represent_strarr_as_indices(base_pair_array)
    freq_array = np.sum(base_pair_index_array == 0, axis = 0)[:,None] # will be (genome_length, num_base_pairs)
    for i in range(1,len(base_pairs)):
        freq_array = np.concatenate((freq_array, np.sum(base_pair_index_array == i, axis = 0)[:,None]), axis = -1)
    rank_array = np.argsort(np.argsort(-1 * freq_array, axis = 1), axis = 1) # (genome_length, num_base_pairs)

    arr = rank_array[np.arange(rank_array.shape[0]), base_pair_index_array]
    return arr

def load_1000genomes_matrix(path='../data/1000genomes/p4dataset2022.txt', representation = 'rank'):
    dataset = load_1000genomes(path)
    labels, base_pair_array = dataset[:,1:3], dataset[:,3:]
    if representation == 'rank':
        matrix = represent_genome_as_rank(base_pair_array)
    elif representation == 'index':
        matrix, _ = represent_strarr_as_indices(base_pair_array)
    else:
        assert False, "{} is not in ['rank', 'index']".format(representation)
    labels = np.array([a + b for a, b in labels])
    return matrix.T, labels

def load_random_matrix(n, d, k, n_groups):
    assert d % n_groups == 0, "d={} must be divisible by n_groups={}".format(d, n_groups)
    U = np.random.normal(size = (n, k))
    group_means = np.random.normal(size = (n_groups, k))
    means = np.repeat(group_means, d // n_groups, axis = 0)
    V = np.random.normal(loc = means, scale = np.ones_like(means), size = (d, k))
    labels = np.repeat(np.arange(n_groups), d // n_groups, axis = 0)
    return U @ V.T, labels
