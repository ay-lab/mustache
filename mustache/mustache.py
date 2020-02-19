import seaborn as sns
import argparse
import os
import sys

from collections import defaultdict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import convolve2d
from statsmodels.stats.multitest import multipletests
import scipy.ndimage.measurements as scipy_measurements
import math


def parseBP(s):
    """
    :param s: string
    :return: string converted to number, taking account for kb or mb
    """
    if not s:
        return False
    if s.isnumeric():
        return int(s)
    s = s.lower()
    if "kb" in s:
        n = s.split("kb")[0]
        if not n.isnumeric():
            return False
        return int(n) * 1000
    elif "mb" in s:
        n = s.split("mb")[0]
        if not n.isnumeric():
            return False
        return int(n) * 1000000
    return False


def parse_args(args):
    parser = argparse.ArgumentParser(description="Check the help flag")

    parser.add_argument("-f",
                        "--file",
                        dest="f_path",
                        help="REQUIRED: Contact map",
                        required=False)
    parser.add_argument("-o",
                        "--outfile",
                        dest="outdir",
                        help="REQUIRED: Name of the output file.\
                       Output is a numpy binary.",
                        required=True)
    parser.add_argument("-r",
                        "--resolution",
                        dest="resolution",
                        help="REQUIRED: Resolution used for the contact maps",
                        required=True)
    parser.add_argument("-bed", "--bed", dest="bed",
                        help="BED file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-m", "--matrix", dest="mat",
                        help="MATRIX file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-b", "--biases", dest="biasfile",
                        help="RECOMMENDED: biases calculated by\
                        ICE or KR norm for each locus for contact map are read from BIASFILE",
                        required=False)
    parser.add_argument(
        "-sz",
        "--sigmaZero",
        dest="s_z",
        type=float,
        default=1.6,
        help="OPTIONAL: sigma0 value for the method. DEFAULT is 1.6. \
        Experimentally chosen for 5Kb resolution",
        required=False)
    parser.add_argument("-oc", "--octaves", dest="octaves", default=2,
                        type=int,
                        help="OPTIONAL: Octave count for the method. \
                        DEFAULT is 2.",
                        required=False)
    parser.add_argument("-i", "--iterations", dest="s", default=10,
                        type=int,
                        help="OPTIONAL: iteration count for the method. \
                        DEFAULT is 10. Experimentally chosen for \
                        5Kb resolution",
                        required=False)
    parser.add_argument(
        "-ch",
        "--chromosome",
        dest="chromosome",
        help="REQUIRED: Specify which chromosome to run the program for.",
        default='n',
        required=True)
    parser.add_argument(
        "-d",
        "--distanceFilter",
        dest="distFilter",
        type=str,
        default='2Mb',
        help="OPTIONAL: If the data is too sparse for distant \
        locations(ie. distance > 2Mb), you can filter them. \
        DEFAULT is 2Mb",
        required=False)
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        type=bool,
                        default=True,
                        help="OPTIONAL: Verbosity of the program",
                        required=False)
    return parser.parse_args()


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def get_diags(map):
    """
    :param map: Contact map, numpy matrix
    :return: 2 Dictionaries where keys are the diagonal number and values are the mean of that diagonal in one dictionary and the std. in the other dictionary.
    """
    means = {}
    stds = {}
    for i in range(len(map)):
        diag = map.diagonal(i)
        diag = diag[diag != 0]
        if len(diag) > 0:
            mean = np.mean(diag)
            std = np.std(diag) if np.std(diag) != 0 else 1
            if math.isnan(mean):
                means[i] = 0
            else:
                means[i] = mean
            if math.isnan(std):
                stds[i] = 1
            else:
                stds[i] = std
        else:
            means[i] = 0
            stds[i] = 1
    return means, stds


def normalize_sparse(x, y, v, cmap, resolution):
    distances = np.abs(y-x)
    size = cmap.shape[1]
    filter_size = int(2_000_000 / resolution)
    for d in range(2 + 2_000_000 // resolution):
        indices = distances == d
        vals = np.zeros(size - d)
        vals[x[indices]] = v[indices]

        vals[x[indices]] = v[indices]+0.001
        if vals.size == 0:
            continue
        std = np.std(v[indices])
        mean = np.mean(v[indices])
        if math.isnan(mean):
            mean = 0
        if math.isnan(std):
            std = 1

        kernel = np.ones(filter_size)
        counts = np.convolve(vals != 0, kernel, mode='same')

        s = np.convolve(vals, kernel, mode='same')
        s2 = np.convolve(vals ** 2, kernel, mode='same')
        local_var = (s2 - s ** 2 / counts) / (counts - 1)

        std2 = std ** 2
        np.nan_to_num(local_var, copy=False,
                      neginf=std2, posinf=std2, nan=std2)

        local_mean = s / counts
        local_mean[counts < 30] = mean
        local_var[counts < 30] = std2

        np.nan_to_num(local_mean, copy=False,
                      neginf=mean, posinf=mean, nan=mean)

        local_std = np.sqrt(local_var)
        vals[x[indices]] -= local_mean[x[indices]]
        vals[x[indices]] /= local_std[x[indices]]
        np.nan_to_num(vals, copy=False, nan=0, posinf=0, neginf=0)
        vals = vals*(1 + math.log(1+mean, 30))
        i = kth_diag_indices(cmap, d)
        cmap[i] = vals


def mustache(cc, chromosome, res, start, end, mask_size, distance, octave_values, outdir):

    c = np.copy(cc[start:end, start:end])

    nz = np.logical_and(c != 0, np.triu(c, 4))
    if np.sum(nz) < 50:
        return []
    c[np.tril_indices_from(c, 4)] = 2
    c[np.triu_indices_from(c, k=(distance+1))] = 2

    pAll = np.ones_like(c[nz]) * 2
    Scales = np.ones_like(pAll)
    s = 10
    curr_filter = 1
    scales = {}
    for o in octave_values:
        print("Octave ", o)
        scales[o] = {}
        sigma = o
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gp = gaussian_filter(c, o, truncate=t, order=0)
        scales[o][1] = sigma

        sigma = o * 2**((2-1)/s)
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gc = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][2] = sigma

        sigma = o * 2**((3-1)/s)
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gn = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][3] = sigma

        Lp = Gp - Gc
        Lc = Gc - Gn

        locMaxP = maximum_filter(
            Lp, footprint=np.ones((3, 3)), mode='constant')
        locMaxC = maximum_filter(
            Lc, footprint=np.ones((3, 3)), mode='constant')
        for i in range(3, s + 2):
            curr_filter += 1
            Gc = Gn

            sigma = o * 2**((i)/s)
            w = 2*math.ceil(2*sigma)+1
            t = ((w - 1)/2 - 0.5)/sigma
            Gn = gaussian_filter(c, sigma, truncate=t, order=0)
            scales[o][i+1] = sigma

            Ln = Gc - Gn
            dist_params = expon.fit(np.abs(Lc[nz]))
            pval = 1 - expon.cdf(np.abs(Lc[nz]), *dist_params)
            locMaxN = maximum_filter(
                Ln, footprint=np.ones((3, 3)), mode='constant')

            willUpdate = np.logical_and \
                .reduce((pval < pAll, Lc[nz] == locMaxC[nz],
                         np.logical_or(Lp[nz] == locMaxP[nz],
                                       Ln[nz] == locMaxN[nz]),
                         Lc[nz] > locMaxP[nz],
                         Lc[nz] > locMaxN[nz]))
            Scales[willUpdate] = scales[o][i]
            pAll[willUpdate] = pval[willUpdate]
            Lp = Lc
            Lc = Ln
            locMaxP = locMaxC
            locMaxC = locMaxN

    pFound = pAll != 2
    if len(pFound) < 10000:
        return []
    _, pCorrect, _, _ = multipletests(pAll[pFound], method='fdr_bh')
    pAll[pFound] = pCorrect

    o = np.ones_like(c)
    o[nz] = pAll
    sig_count = np.sum(o < 0.2)
    x, y = np.unravel_index(np.argsort(o.ravel()), o.shape)
    so = np.ones_like(c)
    so[nz] = Scales

    x = x[:sig_count]
    y = y[:sig_count]
    xyScales = so[x, y]

    nonsparse = x != 0
    for i in range(len(xyScales)):
        s = math.ceil(xyScales[i])
        c1 = np.sum(nz[x[i]-s:x[i]+s+1, y[i]-s:y[i]+s+1]) / \
            ((2*s+1)**2)
        s = 2*s
        c2 = np.sum(nz[x[i]-s:x[i]+s+1, y[i]-s:y[i]+s+1]) / \
            ((2*s+1)**2)
        if c1 < 0.88 or c2 < 0.6:
            nonsparse[i] = False
    x = x[nonsparse]
    y = y[nonsparse]

    if len(x) == 0:
        return []

    def nz_mean(vals):
        return np.mean(vals[vals != 0])

    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])

    means = np.vectorize(diag_mean, excluded=['map'])(k=y-x, map=c)
    passing_indices = c[x, y] > 2*means
    x = x[passing_indices]
    y = y[passing_indices]

    label_matrix = np.zeros((np.max(y)+2, np.max(y)+2), dtype=np.float32)
    label_matrix[x, y] = o[x, y] + 1
    label_matrix[x+1, y] = 2
    label_matrix[x+1, y+1] = 2
    label_matrix[x, y+1] = 2
    label_matrix[x-1, y] = 2
    label_matrix[x-1, y-1] = 2
    label_matrix[x, y-1] = 2
    label_matrix[x+1, y-1] = 2
    label_matrix[x-1, y+1] = 2
    num_features = scipy_measurements.label(
        label_matrix, output=label_matrix, structure=np.ones((3, 3)))

    out = []
    for label in range(1, num_features+1):
        indices = np.argwhere(label_matrix == label)
        i = np.argmin(o[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        out.append([_x+start, _y+start, o[_x, _y], so[_x, _y]])

    print('block done')
    return out


def regulator(f, outdir, bed="",
              res=5000,
              sigma0=1.6,
              s=10,
              octaves=2,
              verbose=True,
              distance_filter=2000000,
              bias=False,
              chromosome='n'):

    octave_values = [sigma0 * (2 ** i) for i in range(octaves)]
    distance = distance_filter

    distance = int(math.ceil(distance // res))
    df = pd.read_csv(f, sep='\t', header=None)
    df = df.loc[np.abs(df[0]-df[1]) <= ((distance+10) * res), :]
    df[0] //= res
    df[1] //= res

    bdf = pd.read_csv(bias, sep='\t', header=None)
    biases = np.nan_to_num(bdf[0], nan=np.Inf)
    biases[biases < 0.2] = np.Inf
    df[2] = df[2] / (biases[df[0]] * biases[df[1]])
    df = df.loc[df[2] > 0, :]

    x = np.min(df.loc[:, [0, 1]], axis=1)
    y = np.max(df.loc[:, [0, 1]], axis=1)

    n = max(y) + 1
    c = np.zeros((n, n), dtype=np.float32)
    c[x, y] = df[2]
    normalize_sparse(x, y, df[2], c, res)
    CHUNK_SIZE = 2000
    overlap_size = 400

    if n <= CHUNK_SIZE:
        start = [0]
        end = [n]
    else:
        start = [0]
        end = [CHUNK_SIZE]

        while end[-1] < n:
            start.append(end[-1]-overlap_size)
            end.append(start[-1] + CHUNK_SIZE)
        end[-1] = n
        start[-1] = end[-1] - CHUNK_SIZE
    o = []
    for i in range(len(start)):
        if i == 0:
            mask_size = -1
        elif i == len(start)-1:
            mask_size = end[i-1] - start[i]
        else:
            mask_size = overlap_size
        loops = mustache(
            c, chromosome, res, start[i], end[i], mask_size, distance, octave_values, outdir)
        for loop in list(loops):
            if loop[0] >= start[i]+mask_size or loop[1] >= start[i]+mask_size:
                o.append([loop[0], loop[1], loop[2], loop[3]])
    return o


def main():

    args = parse_args(sys.argv[1:])
    print("\n")

    f = args.f_path
    if args.bed and args.mat:
        f = args.mat

    if not os.path.exists(f):
        print("Error: Couldn't find specified contact files")
        return
    res = parseBP(args.resolution)
    if not res:
        print("Error: Invalid resolution")
        return
    distFilter = 0

    biasf = False
    if args.biasfile:
        if os.path.exists(args.biasfile):
            biasf = args.biasfile
        else:
            print("Error: Couldn't find specified bias file")
            return

    if args.distFilter:
        distFilter = parseBP(args.distFilter)
        if not distFilter:
            print("Error: Invalid distance filter")
            return
    o = regulator(f, args.outdir,
                  bed=args.bed,
                  res=res,
                  sigma0=args.s_z,
                  s=args.s,
                  verbose=args.verbose,
                  distance_filter=distFilter,
                  bias=biasf,
                  chromosome=args.chromosome,
                  octaves=args.octaves)
    with open(args.outdir, 'w') as out_file:
        for significant in o:
            out_file.write(
                f'{args.chromosome}\t{significant[0]*res}\t{(significant[0]+1)*res}\t{args.chromosome}\t{significant[1]*res}\t{(significant[1]+1)*res}\t{significant[2]}\t{significant[3]}\n')


if __name__ == '__main__':
    main()
