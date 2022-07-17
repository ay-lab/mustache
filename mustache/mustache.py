#!/usr/bin/env python3
import argparse
import os
import sys
import re
import math
import warnings
import time
import struct
from collections import defaultdict

import pandas as pd
import numpy as np
import hicstraw
import cooler

from scipy.stats import expon
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import convolve2d
import scipy.ndimage.measurements as scipy_measurements
from scipy import sparse

from statsmodels.stats.multitest import multipletests

from multiprocessing import Process, Manager


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
    parser.add_argument("-d",
                        "--distance",
                        dest="distFilter",
                        help="REQUIRED: Maximum distance (in bp) allowed between loop loci",
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
        "-cz",
        "--chromosomeSize",
        default="",
        dest="chrSize_file",
        help="RECOMMENDED: .hic corressponfing chromosome size file.",
        required=False)
    parser.add_argument(
        "-norm",
        "--normalization",
        default=False,
        dest="norm_method",
        help="RECOMMENDED: Hi-C  normalization method (KR, VC,...).",
        required=False)
    # parser.add_argument("-cb",
    #                    '--cooler-balance',
    #                     dest='cooler_balance',
    #                     default=False,
    #                     #action='store_false',
    #                     required=False,
    #                     help="OPTIONAL: The cooler data was normalized prior to creating the .cool file.")
    # parser.set_defaults(cooler_balance=False)
    parser.add_argument(
        "-st",
        "--sparsityThreshold",
        dest="st",
        type=float,
        default=0.88,
        help="OPTIONAL: Mustache filters out contacts in sparse areas, you can relax this for sparse datasets(i.e. -st 0.8). Default value is 0.88.",
        required=False)
    parser.add_argument(
        "-pt",
        "--pThreshold",
        dest="pt",
        type=float,
        default=0.2,
        help="OPTIONAL: P-value threshold for the results in the final output. Default is 0.2",
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
    parser.add_argument("-p", "--processes", dest="nprocesses", default=4, type=int,
                        help="OPTIONAL: Number of parallel processes to run. DEFAULT is 4. Increasing this will also increase the memory usage",
                        required=False)
    # parser.add_argument("-c",
    # "--changefile",
    # dest="changedir",
    # help="...",
    # required=False,
    # default="")
    parser.add_argument(
        "-ch",
        "--chromosome",
        dest="chromosome",
        nargs='+',
        help="REQUIRED: Specify which chromosome to run the program for. Optional for cooler files.",
        default='n',
        required=False)
    parser.add_argument(
        "-ch2",
        "--chromosome2",
        dest="chromosome2",
        nargs='+',
        help="Optional: Specify the second chromosome for interchromosomal analysis.",
        default='n',
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


def is_chr(s, c):
    # if 'X' == c or 'chrX':
    #    return 'X' in c
    # if 'Y' == c:
    #    return 'Y' in c
    return str(c).replace('chr', '') == str(s).replace('chr', '')  # re.findall("[1-9][0-9]*", str(s))


def get_sep(f):
    """
    :param f: file path
    :return: Guesses the value separator in the file.
    """
    with open(f) as file:
        for line in file:
            if "\t" in line:
                return '\t'
            if " " in line.strip():
                return ' '
            if "," in line:
                return ','
            if len(line.split(' ')) == 1:
                return ' '
            break
    raise FileNotFoundError


def read_bias(f, chromosome, res):
    """
    :param f: Path to the bias file
    :return: Dictionary where keys are the bin coordinates and values are the bias values.
    """
    d = defaultdict(lambda: 1.0)
    if f:
        sep = get_sep(f)
        with open(f) as file:
            for pos, line in enumerate(file):
                line = line.strip().split(sep)
                if len(line) == 3:
                    if is_chr(line[0], chromosome):
                        val = float(line[2])
                        if not np.isnan(val):
                            if val < 0.2:
                                d[(float(line[1]) // res)] = np.Inf
                            else:
                                d[(float(line[1]) // res)] = val
                        else:
                            d[(float(line[1]) // res)] = np.Inf

                elif len(line) == 1:
                    val = float(line[0])
                    if not np.isnan(val):
                        if val < 0.2:
                            d[pos] = np.Inf
                        else:
                            d[pos] = val
                    else:
                        d[pos] = np.Inf

        return d
    return False


def read_pd(f, distance_in_bp, bias, chromosome, res):
    sep = get_sep(f)
    df = pd.read_csv(f, sep=sep, header=None)
    df.dropna(inplace=True)
    if df.shape[1] == 5:
        df = df[np.vectorize(is_chr)(df[0], chromosome)]
        if df.shape[0] == 0:
            print('Could\'t read any interaction for this chromosome!')
            return
        df = df[np.vectorize(is_chr)(df[2], chromosome)]
        df = df.loc[np.abs(df[1] - df[3]) <= ((distance_in_bp / res + 1) * res), :]
        df[1] //= res
        df[3] //= res
        bias = read_bias(bias, chromosome, res)
        if bias:
            factors = np.vectorize(bias.get)(df[1], 1)
            df[4] = np.divide(df[4], factors)
            factors = np.vectorize(bias.get)(df[3], 1)
            df[4] = np.divide(df[4], factors)

        df = df.loc[df[4] > 0, :]

        x = np.min(df.loc[:, [1, 3]], axis=1)
        y = np.max(df.loc[:, [1, 3]], axis=1)
        val = np.array(df[4])

    elif df.shape[1] == 3:
        df = df.loc[np.abs(df[1] - df[0]) <= ((distance_in_bp / res + 1) * res), :]
        df[0] //= res
        df[1] //= res
        bias = read_bias(bias, chromosome, res)
        if bias:
            factors = np.vectorize(bias.get)(df[0], 1)
            df[2] = np.divide(df[2], factors)
            factors = np.vectorize(bias.get)(df[1], 1)
            df[2] = np.divide(df[2], factors)

        df = df.loc[df[2] > 0, :]

        x = np.min(df.loc[:, [0, 1]], axis=1)
        y = np.max(df.loc[:, [0, 1]], axis=1)
        val = np.array(df[2])

    return x, y, val


def read_hic_file(f, norm_method, CHRM_SIZE, distance_in_bp, chr1, chr2, res):
    """
    :param f: .hic file path
    :param chr: Which chromosome to read the file for
    :param res: Resolution to extract information from
    :return: Numpy matrix of contact counts
    """
    if not CHRM_SIZE:
        hic = hicstraw.HiCFile(f)
        chromosomes = hic.getChromosomes()
        chrSize_in_bp = {}
        for i in range(1, len(chromosomes)):
            chrSize_in_bp["chr" + str(chromosomes[i].name).replace("chr", '')] = chromosomes[i].length
        CHRM_SIZE = chrSize_in_bp["chr" + chr1.replace("chr", '')]

    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE * res)  # CHUNK_SIZE*res
    result = []
    val = []

    while start < CHRM_SIZE:
        print(int(start), int(end))
        if not norm_method:
            temp = hicstraw.straw("observed", "KR", f, str(chr1) + ":" + str(int(start)) + ":" + str(int(end)),
                                  str(chr2) + ":" + str(int(start)) + ":" + str(int(end)), "BP", res)
        else:
            temp = hicstraw.straw("observed", str(norm_method), f,
                                  str(chr1) + ":" + str(int(start)) + ":" + str(int(end)),
                                  str(chr2) + ":" + str(int(start)) + ":" + str(int(end)), "BP", res)
        if len(temp) == 0:
            start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
            if end == CHRM_SIZE - 1:
                break
            else:
                end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
            continue

        if result == []:
            cur_block = [[int(record.binX), int(record.binY), record.counts] for record in temp]
            result.append([x[0] for x in cur_block])
            result.append([x[1] for x in cur_block])
            result.append([x[2] for x in cur_block])            
            prev_block = set([(record.binX, record.binY, record.counts) for record in temp])
        else:
            cur_block = set([(int(record.binX), int(record.binY), record.counts) for record in temp])
            to_add_list = list(cur_block - prev_block)
            del prev_block
            result[0] += [x[0] for x in to_add_list]
            result[1] += [x[1] for x in to_add_list]
            result[2] += [x[2] for x in to_add_list]
            prev_block = cur_block
            del cur_block
        start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
        if end == CHRM_SIZE - 1:
            break
        else:
            end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)

    x = np.array(result[0]) // res
    y = np.array(result[1]) // res
    val = np.array(result[2])

    nan_indx = np.logical_or.reduce((np.isnan(result[0]), np.isnan(result[1]), np.isnan(result[2])))
    x = x[~nan_indx]
    y = y[~nan_indx]
    val = val[~nan_indx]
    x = x.astype(int)
    y = y.astype(int)

    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []
    else:
        val[np.isnan(val)] = 0

    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    if len(val > 0):
        return x, y, val
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []


def read_cooler(f, distance_in_bp, chr1, chr2, cooler_balance):
    """
    :param f: .cool file path
    :param chr: Which chromosome to read the file for
    :return: Numpy matrix of contact counts
    """
    clr = cooler.Cooler(f)
    res = clr.binsize
    print(f'Your cooler data resolution is {res}')
    if chr1 not in clr.chromnames or chr2 not in clr.chromnames:
        raise NameError('wrong chromosome name!')
    CHRM_SIZE = clr.chromsizes[chr1]
    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHUNK_SIZE * res, CHRM_SIZE)  # CHUNK_SIZE*res
    result = []
    val = []
    ###########################
    if chr1 == chr2:
        # try:
        # normVec = clr.bins()['weight'].fetch(chr1)
        # result = clr.matrix(balance=True,sparse=True).fetch(chr1)#as_pixels=True, join=True
        while start < CHRM_SIZE:
            print(int(start), int(end))
            if not cooler_balance:
                temp = clr.matrix(balance=True, sparse=True).fetch((chr1, int(start), int(end)))
            else:
                temp = clr.matrix(balance=cooler_balance, sparse=True).fetch((chr1, int(start), int(end)))
            temp = sparse.triu(temp)
            np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
            start_in_px = int(start / res)
            if len(temp.row) == 0:
                start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                if end == CHRM_SIZE - 1:
                    break
                else:
                    end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
                continue

            if result == []:
                result += [list(start_in_px + temp.row), list(start_in_px + temp.col), list(temp.data)]
                prev_block = set(
                    [(x, y, v) for x, y, v in zip(start_in_px + temp.row, start_in_px + temp.col, temp.data)])
            else:
                cur_block = set(
                    [(x, y, v) for x, y, v in zip(start_in_px + temp.row, start_in_px + temp.col, temp.data)])
                to_add_list = list(cur_block - prev_block)
                del prev_block
                result[0] += [x[0] for x in to_add_list]
                result[1] += [x[1] for x in to_add_list]
                result[2] += [x[2] for x in to_add_list]
                prev_block = cur_block
                del cur_block

            start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
            if end == CHRM_SIZE - 1:
                break
            else:
                end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
                # except:
        # raise NameError('Reading from the file failed!')
        if len(result) == 0:
            print(f'There is no contact in chrmosome {chr1} to work on.')
            return [], [], [], res

        x = np.array(result[0])
        y = np.array(result[1])
        val = np.array(result[2])
    else:

        result = clr.matrix(balance=True, sparse=True).fetch(chr1, chr2)
        result = sparse.triu(result)
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        x = result.row
        y = result.col
        val = result.data

    ##########################
    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], [], res
    else:
        val[np.isnan(val)] = 0

    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]
    # return np.array(x),np.array(y),np.array(val), res, normVec
    if len(val > 0):
        return np.array(x), np.array(y), np.array(val), res
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], [], res


def read_mcooler(f, distance_in_bp, chr1, chr2, res, cooler_balance):
    """
    :param f: .cool file path
    :param chr: Which chromosome to read the file for
    :param res: Resolution to extract information from
    :return: Numpy matrix of contact counts
    """
    uri = '%s::/resolutions/%s' % (f, res)
    # uri = '%s::/7' % (f)
    clr = cooler.Cooler(uri)
    # print(clr.bins()[:100])
    if chr1 not in clr.chromnames or chr2 not in clr.chromnames:
        raise NameError('wrong chromosome name!')
    CHRM_SIZE = clr.chromsizes[chr1]
    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE * res)  # CHUNK_SIZE*res
    result = []
    val = []

    if chr1 == chr2:
        try:
            # result = clr.matrix(balance=True,sparse=True).fetch(chr1)#as_pixels=True, join=True
            while start < CHRM_SIZE:
                print(int(start), int(end))
                if not cooler_balance:
                    temp = clr.matrix(balance=True, sparse=True).fetch((chr1, int(start), int(end)))
                else:
                    temp = clr.matrix(balance=cooler_balance, sparse=True).fetch((chr1, int(start), int(end)))
                temp = sparse.triu(temp)
                np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
                start_in_px = int(start / res)
                if len(temp.row) == 0:
                    start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                    if end == CHRM_SIZE - 1:
                        break
                    else:
                        end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)

                    continue

                if result == []:
                    result += [list(start_in_px + temp.row), list(start_in_px + temp.col), list(temp.data)]
                    prev_block = set(
                        [(x, y, v) for x, y, v in zip(start_in_px + temp.row, start_in_px + temp.col, temp.data)])
                    # print('result==[]')
                else:
                    cur_block = set(
                        [(x, y, v) for x, y, v in zip(start_in_px + temp.row, start_in_px + temp.col, temp.data)])
                    to_add_list = list(cur_block - prev_block)
                    del prev_block
                    result[0] += [x[0] for x in to_add_list]
                    result[1] += [x[1] for x in to_add_list]
                    result[2] += [x[2] for x in to_add_list]
                    prev_block = cur_block
                    del cur_block

                start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                if end == CHRM_SIZE - 1:
                    break
                else:
                    end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
        except:
            raise NameError('Reading from the file failed!')

        if len(result) == 0:
            print(f'There is no contact in chrmosome {chr1} to work on.')
            return [], [], []

        x = np.array(result[0])
        y = np.array(result[1])
        val = np.array(result[2])
    else:
        result = clr.matrix(balance=True, sparse=True).fetch(chr1, chr2)
        result = sparse.triu(result)
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        x = result.row
        y = result.col
        val = result.data

    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []
    else:
        val[np.isnan(val)] = 0

    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    if len(val > 0):
        return np.array(x), np.array(y), np.array(val)
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []


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


def normalize_sparse(x, y, v, resolution, distance_in_px):
    n = max(max(x), max(y)) + 1

    # distance_in_px = min(distance_in_px, n)
    pval_weights = []
    distances = np.abs(y - x)
    if (n - distance_in_px) * resolution > 2000000:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            filter_size = int(2000000 / resolution)
            for d in range(2 + distance_in_px):
                indices = distances == d
                vals = np.zeros(n - d)
                vals[x[indices]] = v[indices] + 0.001
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
                vals = vals * (1 + math.log(1 + mean, 30))
                pval_weights += [1 + math.log(1 + mean, 30)]
                v[indices] = vals[x[indices]]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            np.nan_to_num(v, copy=False, neginf=0, posinf=0, nan=0)
            distance_in_px = min(distance_in_px, n)
            for d in range(distance_in_px):
                indices = distances == d
                std = np.std(v[indices])
                mean = np.mean(v[indices])
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1
                # print(std)
                v[indices] = (v[indices] - mean) / std
                np.nan_to_num(v, copy=False, nan=0, posinf=0, neginf=0)
    return pval_weights


def inter_nrmalize_map(vals):
    m = np.mean(vals)
    s = np.std(vals)
    cmap -= m
    cmap /= s
    np.nan_to_num(cmap, copy=False, nan=0, posinf=0, neginf=0)


def mustache(c, chromosome, chromosome2, res, pval_weights, start, end, mask_size, distance_in_px, octave_values, st,
             pt):
    nz = np.logical_and(c != 0, np.triu(c, 4))
    nz_temp = np.logical_and.reduce((c != 0, np.triu(c, 4) > 0, np.tril(c, distance_in_px) > 0))
    if np.sum(nz) < 50:
        return []
    c[np.tril_indices_from(c, 4)] = 2

    if chromosome == chromosome2:
        c[np.triu_indices_from(c, k=(distance_in_px + 1))] = 2

    pAll = np.ones_like(c[nz]) * 2
    Scales = np.ones_like(pAll)
    vAll = np.zeros_like(pAll)
    s = 10
    # curr_filter = 1
    scales = {}
    for o in octave_values:
        scales[o] = {}
        sigma = o
        w = 2 * math.ceil(2 * sigma) + 1
        t = (((w - 1) / 2) - 0.5) / sigma
        Gp = gaussian_filter(c, o, truncate=t, order=0)
        scales[o][1] = sigma

        sigma = o * 2 ** ((2 - 1) / s)
        w = 2 * math.ceil(2 * sigma) + 1
        t = (((w - 1) / 2) - 0.5) / sigma
        Gc = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][2] = sigma

        Lp = Gp - Gc
        Gp = []

        sigma = o * 2 ** ((3 - 1) / s)
        w = 2 * math.ceil(2 * sigma) + 1
        t = (((w - 1) / 2) - 0.5) / sigma
        Gn = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][3] = sigma

        # Lp = Gp - Gc
        Lc = Gc - Gn

        locMaxP = maximum_filter(
            Lp, footprint=np.ones((3, 3)), mode='constant')
        locMaxC = maximum_filter(
            Lc, footprint=np.ones((3, 3)), mode='constant')
        for i in range(3, s + 2):
            # curr_filter += 1
            Gc = Gn

            sigma = o * 2 ** ((i) / s)
            w = 2 * math.ceil(2 * sigma) + 1
            t = ((w - 1) / 2 - 0.5) / sigma
            Gn = gaussian_filter(c, sigma, truncate=t, order=0)
            scales[o][i + 1] = sigma

            Ln = Gc - Gn
            dist_params = expon.fit(np.abs(Lc[nz]))
            pval = 1 - expon.cdf(np.abs(Lc[nz]), *dist_params)
            locMaxN = maximum_filter(
                Ln, footprint=np.ones((3, 3)), mode='constant')

            willUpdate = np.logical_and \
                .reduce((Lc[nz] > vAll, Lc[nz] == locMaxC[nz],
                         np.logical_or(Lp[nz] == locMaxP[nz],
                                       Ln[nz] == locMaxN[nz]),
                         Lc[nz] > locMaxP[nz],
                         Lc[nz] > locMaxN[nz]))
            vAll[willUpdate] = Lc[nz][willUpdate]
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

    #################
    # o = np.ones_like(c)
    # o[nz] = pAll
    # x, y = np.where(nz_temp)
    # o[x,y]*=np.array(pval_weights)[y-x]
    # o[x,y]/=10
    # pAll = o[nz]
    #################
    o = np.ones_like(c)
    o[nz] = pAll
    sig_count = np.sum(o < pt)  # change
    x, y = np.unravel_index(np.argsort(o.ravel()), o.shape)
    so = np.ones_like(c)
    so[nz] = Scales

    x = x[:sig_count]
    y = y[:sig_count]
    xyScales = so[x, y]

    nonsparse = x != 0
    for i in range(len(xyScales)):
        s = math.ceil(xyScales[i])
        c1 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / \
             ((2 * s + 1) ** 2)
        s = 2 * s
        c2 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / \
             ((2 * s + 1) ** 2)
        if c1 < st or c2 < 0.6:
            nonsparse[i] = False
    x = x[nonsparse]
    y = y[nonsparse]

    if len(x) == 0:
        return []

    def nz_mean(vals):
        return np.mean(vals[vals != 0])

    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])

    if chromosome == chromosome2:
        means = np.vectorize(diag_mean, excluded=['map'])(k=y - x, map=c)
        passing_indices = c[x, y] > 2 * means  # change
        if len(passing_indices) == 0 or np.sum(passing_indices) == 0:
            return []
        x = x[passing_indices]
        y = y[passing_indices]

    label_matrix = np.zeros((np.max(y) + 2, np.max(y) + 2), dtype=np.float32)
    label_matrix[x, y] = o[x, y] + 1
    label_matrix[x + 1, y] = 2
    label_matrix[x + 1, y + 1] = 2
    label_matrix[x, y + 1] = 2
    label_matrix[x - 1, y] = 2
    label_matrix[x - 1, y - 1] = 2
    label_matrix[x, y - 1] = 2
    label_matrix[x + 1, y - 1] = 2
    label_matrix[x - 1, y + 1] = 2
    num_features = scipy_measurements.label(
        label_matrix, output=label_matrix, structure=np.ones((3, 3)))

    out = []
    for label in range(1, num_features + 1):
        indices = np.argwhere(label_matrix == label)
        i = np.argmin(o[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        out.append([_x + start, _y + start, o[_x, _y], so[_x, _y]])

    return out


def regulator(f, norm_method, CHRM_SIZE, outdir, bed="",
              res=5000,
              sigma0=1.6,
              s=10,
              pt=0.1,
              st=0.88,
              octaves=2,
              verbose=True,
              nprocesses=4,
              distance_filter=2000000,
              bias=False,
              chromosome='n',
              chromosome2=None):
    if not chromosome2 or chromosome2 == 'n':
        chromosome2 = chromosome

    if (chromosome != chromosome2) and not ((('.hic' in f) or ('.cool' in f) or ('.mcool' in f))):
        print(
            "Interchromosomal analysis is only supported for .hic and .cool input formats.")
        raise FileNotFoundError

    octave_values = [sigma0 * (2 ** i) for i in range(octaves)]
    distance_in_bp = distance_filter

    print("Reading contact map...")

    if f.endswith(".hic"):
        x, y, v = read_hic_file(f, norm_method, CHRM_SIZE, distance_in_bp, chromosome, chromosome2, res)
    elif f.endswith(".cool"):
        x, y, v, res = read_cooler(f, distance_in_bp, chromosome, chromosome2, norm_method)
    elif f.endswith(".mcool"):
        x, y, v = read_mcooler(f, distance_in_bp, chromosome, chromosome2, res, norm_method)
    else:
        x, y, v = read_pd(f, distance_in_bp, bias, chromosome, res)

    if len(v) == 0:
        return []
    print("Normalizing contact map...")

    distance_in_px = int(math.ceil(distance_in_bp // res))
    if chromosome == chromosome2:
        n = max(max(x), max(y)) + 1
        pval_weights = normalize_sparse(x, y, v, res, distance_in_px)
        CHUNK_SIZE = max(2 * distance_in_px, 2000)
        overlap_size = distance_in_px

        if n <= CHUNK_SIZE:
            start = [0]
            end = [n]
        else:
            start = [0]
            end = [CHUNK_SIZE]

            while end[-1] < n:
                start.append(end[-1] - overlap_size)
                end.append(start[-1] + CHUNK_SIZE)
            end[-1] = n
            start[-1] = end[-1] - CHUNK_SIZE

        print("Loop calling...")
        with Manager() as manager:
            o = manager.list()
            i = 0
            processes = []
            for i in range(len(start)):
                # create the currnet block
                indx = np.logical_and.reduce((x >= start[i], x < end[i], y >= start[i], y < end[i]))
                xc = x[indx] - start[i]
                yc = y[indx] - start[i]
                vc = v[indx]
                cc = np.zeros((CHUNK_SIZE, CHUNK_SIZE))
                cc[xc, yc] = vc
                #             
                p = Process(target=process_block, args=(
                    i, start, end, overlap_size, cc, chromosome, chromosome2, res, pval_weights, distance_in_px,
                    octave_values, o, st, pt))
                p.start()
                processes.append(p)
                if len(processes) >= nprocesses or i == (len(start) - 1):
                    for p in processes:
                        p.join()
                    processes = []
            # o_corrected = [[e[0],e[1],e[2]/pval_weights[e[1]-e[0]],e[3]] for e in list(o)]

            return list(o)

    else:  # interchromosomal
        n1 = max(x) + 1
        n2 = max(y) + 1
        inter_normalize_map(x, y, v, res)


def process_block(i, start, end, overlap_size, cc, chromosome, chromosome2, res, pval_weights, distance_in_px,
                  octave_values, o, st, pt):
    print("Starting block ", i + 1, "/", len(start), "...", sep='')
    if i == 0:
        mask_size = -1
    elif i == len(start) - 1:
        mask_size = end[i - 1] - start[i]
    else:
        mask_size = overlap_size
    loops = mustache(
        cc, chromosome, chromosome2, res, pval_weights, start[i], end[i], mask_size, distance_in_px, octave_values, st,
        pt)
    for loop in list(loops):
        if loop[0] >= start[i] + mask_size or loop[1] >= start[i] + mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3]])
    print("Block", i + 1, "done.")


def main():
    start_time = time.time()
    args = parse_args(sys.argv[1:])
    print("\n")

    f = args.f_path
    if args.bed and args.mat:
        f = args.mat

    if not os.path.exists(f):
        print("Error: Couldn't find the specified contact files")
        return
    res = parseBP(args.resolution)
    if not res:
        print("Error: Invalid resolution")
        return
    CHR_LIST_FLAG = False
    CHR_COOL_FLAG = False
    CHR_HIC_FLAG = False
    if not args.chromosome or args.chromosome == 'n':
        if f.endswith(".cool") or f.endswith(".mcool"):
            CHR_COOL_FLAG = True
        elif f.endswith(".hic"):
            CHR_HIC_FLAG = True
        elif len(args.chromosome > 1):
            print("Error: For this data type you should enter only one chromosome name.")
            return
        else:
            print("Error: Please enter the chromosome name.")
            return
    elif len(args.chromosome) > 1:
        CHR_LIST_FLAG = True

    distFilter = parseBP(args.distFilter)  # change
    if not distFilter:
        if 200 * res >= 2000000:
            distFilter = 200 * res
            print("The distance limit is set to {}bp".format(200 * res))
        elif 2000 * res <= 2000000:
            distFilter = 2000 * res
            print("The distance limit is set to {}bp".format(2000 * res))
        else:
            distFilter = 2000000
            print("The distance limit is set to 2Mbp")
    elif distFilter < 200 * res:
        print("The distance limit is set to {}bp".format(200 * res))
        distFilter = 200 * res
    elif distFilter > 2000 * res:
        print("The distance limit is set to {}bp".format(2000 * res))
        distFilter = 2000 * res
    elif distFilter > 2000000:
        distFilter = 2000000
        print("The distance limit is set to 2Mbp")

    # distFilter = 4000000
    chrSize_in_bp = False
    if CHR_COOL_FLAG:
        # extract all the chromosome names big enough to run mustache on
        chr_list = []
        if f.endswith(".cool"):
            clr = cooler.Cooler(f)
        else:  # mcooler
            uri = '%s::/resolutions/%s' % (f, res)
            clr = cooler.Cooler(uri)
        for i, chrm in enumerate(clr.chromnames):
            if clr.chromsizes[i] > 1000000:
                chr_list.append(chrm)
    elif CHR_HIC_FLAG:
        hic = hicstraw.HiCFile(f)
        chromosomes = hic.getChromosomes()
        chr_list = [chromosomes[i].name for i in range(1, len(chromosomes))]
        chrSize_in_bp = {}
        for i in range(1, len(chromosomes)):
            chrSize_in_bp["chr" + str(chromosomes[i].name).replace("chr", '')] = chromosomes[i].length
    else:
        chr_list = args.chromosome.copy()

    if (args.chromosome2 and args.chromosome2 != 'n') and (len(chr_list) != len(args.chromosome2)):
        print("Error: the same number of chromosome1 and chromosome2 should be provided.")
        return
    elif type(args.chromosome2) == list:
        chr_list2 = args.chromosome2.copy()
    else:
        chr_list2 = chr_list.copy()

    CHRM_SIZE = False
    if args.chrSize_file and (not chrSize_in_bp):
        csz_file = args.chrSize_file
        csz = pd.read_csv(csz_file, header=None, sep='\t')
        chrSize_in_bp = {}
        for i in range(csz.shape[0]):
            chrSize_in_bp["chr" + str(csz.iloc[i, 0]).replace('chr', '')] = csz.iloc[i, 1]

    first_chr_to_write = True
    for i, (chromosome, chromosome2) in enumerate(zip(chr_list, chr_list2)):
        if chrSize_in_bp:
            CHRM_SIZE = chrSize_in_bp["chr" + str(chromosome).replace('chr', '')]
        biasf = False
        if args.biasfile:
            if os.path.exists(args.biasfile):
                biasf = args.biasfile
            else:
                print("Error: Couldn't find specified bias file")
                return
        o = regulator(f, args.norm_method, CHRM_SIZE, args.outdir,
                      bed=args.bed,
                      res=res,
                      sigma0=args.s_z,
                      s=args.s,
                      verbose=args.verbose,
                      pt=args.pt,
                      st=args.st,
                      distance_filter=distFilter,
                      nprocesses=args.nprocesses,
                      bias=biasf,
                      chromosome=chromosome,
                      chromosome2=chromosome2,
                      octaves=args.octaves)
        if i == 0:
            with open(args.outdir, 'w') as out_file:
                out_file.write(
                    "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
        if o == []:
            print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o), chromosome, args.pt,
                                                                                "%.2f" % (time.time() - start_time)))
            start_time = time.time()
            continue

        # if first_chr_to_write:
        #    first_chr_to_write = False
        print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o), chromosome, args.pt,
                                                                            "%.2f" % (time.time() - start_time)))

        with open(args.outdir, 'a') as out_file:
            # out_file.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
            for significant in o:
                out_file.write(
                    str(chromosome) + '\t' + str(significant[0] * res) + '\t' + str((significant[0] + 1) * res) + '\t' +
                    str(chromosome2) + '\t' + str(significant[1] * res) + '\t' + str(
                        (significant[1] + 1) * res) + '\t' + str(significant[2]) +
                    '\t' + str(significant[3]) + '\n')
        # else:
        #    print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o),chromosome,args.pt,"%.2f" % (time.time()-old_time)))
        #    with open(args.outdir, 'a') as out_file:
        #        for significant in o:
        #            out_file.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
        #	                   str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
        #		           '\t' + str(significant[3]) + '\n')
        start_time = time.time()


if __name__ == '__main__':
    main()
