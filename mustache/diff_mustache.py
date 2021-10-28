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
import straw
import cooler

from scipy.stats import expon, norm
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

    parser.add_argument("-f1",
                        "--file1",
                        dest="f_path1",
                        help="REQUIRED: Contact map",
                        required=False)
    parser.add_argument("-f2",
                        "--file2",
                        dest="f_path2",
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
    parser.add_argument("-bed1", "--bed1", dest="bed1",
                        help="BED file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-m1", "--matrix1", dest="mat1",
                        help="MATRIX file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-b1", "--biases1", dest="biasfile1",
                        help="RECOMMENDED: biases calculated by\
                        ICE or KR norm for each locus for contact map are read from BIASFILE",
                        required=False)
    parser.add_argument("-bed2", "--bed2", dest="bed2",
                        help="BED file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-m2", "--matrix2", dest="mat2",
                        help="MATRIX file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-b2", "--biases2", dest="biasfile2",
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
    #parser.add_argument("-cb",
    #                    '--cooler-balance',
    #                     dest='cooler_balance',
    #                     default=False,
    #                     #action='store_false',
    #                     required=False,
    #                     help="OPTIONAL: The cooler data was normalized prior to creating the .cool file.")
    #parser.set_defaults(cooler_balance=False)
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
        "-pt2",
        "--pThreshold2",
        dest="pt2",
        type=float,
        default=0.1,
        help="OPTIONAL: P-value threshold for the results in the final output. Default is 0.1",
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
                        help="OPTIONAL: Number of parallel processes to run. DEFAULT is 4. Increasing this will also increase the memory usage", required=False)
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
    #if 'X' == c or 'chrX':
    #    return 'X' in c
    #if 'Y' == c:
    #    return 'Y' in c
    return str(c).replace('chr','') == str(s).replace('chr','')#re.findall("[1-9][0-9]*", str(s))

def readcstr(f):
    #*    Title: hic2cool
    #*    Author: Carl Vitzthum
    #*    Date: 2021
    #*    Code version: 0.8.3
    #*    Availability: https://github.com/4dn-dcic/hic2cool/blob/master/hic2cool/hic2cool_utils.py

    # buf = bytearray()
    buf = b""
    while True:
        b = f.read(1)
        if b is None or b == b"\0":
            # return buf.encode("utf-8", errors="ignore")
            return buf.decode("utf-8")
        elif b == "":
            raise EOFError("Buffer unexpectedly empty while trying to read null-terminated string")
        else:
            buf += b

def read_header(req):
    #*    Title: hic2cool
    #*    Author: Carl Vitzthum
    #*    Date: 2021
    #*    Code version: 0.8.3
    #*    Availability: https://github.com/4dn-dcic/hic2cool/blob/master/hic2cool/hic2cool_utils.py
    """
    Takes in a .hic file and returns a dictionary containing information about
    the chromosome. Keys are chromosome index numbers (0 through # of chroms
    contained in file) and values are [chr idx (int), chr name (str), chrom
    length (str)]. Returns the masterindex used by the file as well as the open
    file object.
    """
    chrs = {}
    resolutions = []
    magic_string = struct.unpack(b'<3s', req.read(3))[0]
    req.read(1)
    if (magic_string != b"HIC"):
        error_string = ('... This does not appear to be a HiC file; '
                       'magic string is incorrect')
        force_exit(error_string, req)
    global version
    version = struct.unpack(b'<i', req.read(4))[0]
    masterindex = struct.unpack(b'<q', req.read(8))[0]
    genome = b""
    c = req.read(1)
    while (c != b'\0'):
        genome += c
        c = req.read(1)
    genome = genome.decode('ascii')
    # metadata extraction
    metadata = {}
    nattributes = struct.unpack(b'<i', req.read(4))[0]
    for x in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        metadata[key] = value
    nChrs = struct.unpack(b'<i', req.read(4))[0]
    for i in range(0, nChrs):
        name = readcstr(req)
        length = struct.unpack(b'<i', req.read(4))[0]
        if name and length:
            chrs[i] = [i, name, length]
    nBpRes = struct.unpack(b'<i', req.read(4))[0]
    # find bp delimited resolutions supported by the hic file
    for x in range(0, nBpRes):
        res = struct.unpack(b'<i', req.read(4))[0]
        resolutions.append(res)
    return chrs, resolutions, masterindex, genome, metadata


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
            if len(line.split(' '))==1:
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
            for pos,line in enumerate(file):
                line = line.strip().split(sep)
                if len(line)==3:
                    if is_chr(line[0], chromosome):
                        val = float(line[2])
                        if not np.isnan(val):
                            if val<0.2:
                                d[(float(line[1]) // res)] = np.Inf
                            else:
                                d[(float(line[1]) // res)] = val
                        else:
                            d[(float(line[1]) // res)] = np.Inf
							
                elif len(line)==1:
                    val = float(line[0])
                    if not np.isnan(val):
                        if val<0.2:
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
    if df.shape[1]==5:       
        df = df[np.vectorize(is_chr)(df[0], chromosome)]
        if df.shape[0] == 0:
            print('Could\'t read any interaction for this chromosome!')
            return 
        df = df[np.vectorize(is_chr)(df[2], chromosome)]
        df = df.loc[np.abs(df[1]-df[3]) <= ((distance_in_bp/res+1) * res), :]
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
    
    elif df.shape[1]==3:
        df = df.loc[np.abs(df[1]-df[0]) <= ((distance_in_bp/res+1) * res), :]
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
    
    return x,y,val

def read_hic_file(f, norm_method, CHRM_SIZE,  distance_in_bp, chr1, chr2, res):
    """
    :param f: .hic file path
    :param chr: Which chromosome to read the file for
    :param res: Resolution to extract information from
    :return: Numpy matrix of contact counts
    """
    if not CHRM_SIZE:
        hic = open(f, 'rb')
        chrs, resolutions, masterindex, genome, metadata = read_header(hic)
        #chr_list = [chrs[i][1] for i in range(1,len(chrs))]         
        chrSize_in_bp = {}
        for i in range(1,len(chrs)):
            chrSize_in_bp["chr"+chrs[i][1].replace("chr",'')] = chrs[i][2]
        CHRM_SIZE = chrSize_in_bp["chr"+chr1.replace("chr",'')]

    
    CHUNK_SIZE = max(2*distance_in_bp/res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE*res) #CHUNK_SIZE*res
    result = []
    val = []
    #try: 
    straw_ver = int(str(straw.__version__).replace('.',''))
    while start < CHRM_SIZE:
        print(int(start),int(end))
        if not norm_method:
            if straw_ver < 1000:
                temp = straw.straw("KR", f, str(chr1)+":"+str(int(start))+":"+str(int(end)),  str(chr2)+":"+str(int(start))+":"+str(int(end)), "BP", res) 
            else:
                temp = straw.straw("observed","KR", f, str(chr1)+":"+str(int(start))+":"+str(int(end)),  str(chr2)+":"+str(int(start))+":"+str(int(end)), "BP", res)
        else:
            if straw_ver < 1000:
                temp = straw.straw(str(norm_method), f, str(chr1)+":"+str(int(start))+":"+str(int(end)),  str(chr2)+":"+str(int(start))+":"+str(int(end)), "BP", res)
            else:
                temp = straw.straw("observed",str(norm_method), f, str(chr1)+":"+str(int(start))+":"+str(int(end)),  str(chr2)+":"+str(int(start))+":"+str(int(end)), "BP", res)
        if len(temp)==0:
            start = min( start + CHUNK_SIZE*res -  distance_in_bp, CHRM_SIZE)
            if end==CHRM_SIZE-1:
                break
            else:
                end = min(end + CHUNK_SIZE*res - distance_in_bp, CHRM_SIZE-1)
            continue
 
        if result == []:
                                         
            if straw_ver < 1000:
                result+=temp
                prev_block = set([(x,y,v) for x,y,v in zip(temp[0],temp[1],temp[2])])
            else:
                result+= [[int(record.binX), int(record.binY), record.counts] for record in temp]
                prev_block = set([(record.binX, record.binY, record.counts) for record in temp])
        else:
            if straw_ver < 1000:
                cur_block = set([(x,y,v) for x,y,v in zip(temp[0],temp[1],temp[2])])
            else:
                cur_block = set([(int(record.binX), int(record.binY), record.counts) for record in temp])
            
            to_add_list = list(cur_block - prev_block)
            del prev_block
            result[0]+=  [x[0] for x in  to_add_list]
            result[1]+=  [x[1] for x in  to_add_list]
            result[2]+=  [x[2] for x in  to_add_list]
            prev_block = cur_block
            del cur_block
        start = min( start + CHUNK_SIZE*res -  distance_in_bp, CHRM_SIZE)
        if end==CHRM_SIZE-1:
            break
        else:
            end = min(end + CHUNK_SIZE*res - distance_in_bp, CHRM_SIZE-1)
    
    x = np.array(result[0]) // res
    y = np.array(result[1]) // res
    val = np.array(result[2])
    nan_indx = np.logical_or.reduce((np.isnan(result[0]),np.isnan(result[1]),np.isnan(result[2])))
    x = x[~nan_indx]
    y = y[~nan_indx]
    val = val[~nan_indx]
    x = x.astype(int)
    y = y.astype(int)
 
    if len(val)==0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [],[],[],res
    else:
        val[np.isnan(val)] = 0

    

    if(chr1==chr2):
        dist_f = np.logical_and(np.abs(x-y) <= distance_in_bp/res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    if len(val>0):
        return x, y, val
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []
	
def read_mcooler(f, distance_in_bp, chr1, chr2, res, cooler_balance):
    """
    :param f: .cool file path
    :param chr: Which chromosome to read the file for
    :param res: Resolution to extract information from
    :return: Numpy matrix of contact counts
    """
    uri = '%s::/resolutions/%s' % (f, res)
    #uri = '%s::/7' % (f)
    clr = cooler.Cooler(uri)
    #print(clr.bins()[:100])
    if chr1 not in clr.chromnames or chr2 not in clr.chromnames:
        raise NameError('wrong chromosome name!')
    CHRM_SIZE = clr.chromsizes[chr1]    
    CHUNK_SIZE = max(2*distance_in_bp/res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE*res) #CHUNK_SIZE*res
    result = []
    val = []
	
    if chr1 == chr2:
        try:
            #result = clr.matrix(balance=True,sparse=True).fetch(chr1)#as_pixels=True, join=True
            while start < CHRM_SIZE:
                print(int(start),int(end))               
                if not cooler_balance: 
                    temp = clr.matrix(balance=True,sparse=True).fetch( (chr1, int(start), int(end)))
                else:
                    temp = clr.matrix(balance=cooler_balance,sparse=True).fetch( (chr1, int(start), int(end)))
                temp = sparse.triu(temp)
                np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
                start_in_px = int(start/res)
                if len(temp.row)==0:
                    start = min( start + CHUNK_SIZE*res -  distance_in_bp, CHRM_SIZE)
                    if end==CHRM_SIZE-1:
                        break
                    else:
                        end = min(end + CHUNK_SIZE*res - distance_in_bp, CHRM_SIZE-1)

                    continue
           
                if result == []:
                    result+= [list(start_in_px+temp.row),list(start_in_px+temp.col),list(temp.data)] 
                    prev_block = set([(x,y,v) for x,y,v in zip(start_in_px+temp.row,start_in_px+temp.col,temp.data)])
                    #print('result==[]')
                else:
                    cur_block = set([(x,y,v) for x,y,v in zip(start_in_px+temp.row,start_in_px+temp.col,temp.data)])
                    to_add_list = list(cur_block - prev_block)
                    del prev_block
                    result[0]+=  [x[0] for x in  to_add_list]
                    result[1]+=  [x[1] for x in  to_add_list]
                    result[2]+=  [x[2] for x in  to_add_list]
                    prev_block = cur_block
                    del cur_block

                start = min( start + CHUNK_SIZE*res -  distance_in_bp, CHRM_SIZE)
                if end==CHRM_SIZE-1:
                    break
                else:
                    end = min(end + CHUNK_SIZE*res - distance_in_bp, CHRM_SIZE-1)
        except:
            raise NameError('Reading from the file failed!')
        
        if len(result)==0:
            print(f'There is no contact in chrmosome {chr1} to work on.')
            return [],[],[]

        x = np.array(result[0])
        y = np.array(result[1])
        val = np.array(result[2])
    else:
        result = clr.matrix(balance=True,sparse=True).fetch(chr1, chr2)    
        result = sparse.triu(result)
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        x = result.row
        y = result.col
        val = result.data
    
    if len(val)==0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [],[],[]
    else:
        val[np.isnan(val)] = 0

    if(chr1==chr2):
        dist_f = np.logical_and(np.abs(x-y) <= distance_in_bp/res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    if len(val>0):
        return np.array(x),np.array(y),np.array(val)
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
    n = max(max(x),max(y)) + 1

    #distance_in_px = min(distance_in_px, n)
    distances = np.abs(y-x)
    if (n-distance_in_px)*resolution  > 2000000: 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            filter_size = int(2000000 / resolution)
            for d in range(2 + distance_in_px):
                indices = distances == d
                vals = np.zeros(n-d)
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
            #print(std)
                v[indices] = (v[indices] - mean)/std
                np.nan_to_num(v, copy=False, nan=0, posinf=0, neginf=0) 
			
def inter_nrmalize_map(vals):
    m = np.mean(vals)
    s = np.std(vals)
    cmap -= m
    cmap /= s
    np.nan_to_num(cmap, copy=False, nan=0, posinf=0, neginf=0)


def diff_mustache(c1, c2, chromosome,chromosome2, res, start, end, mask_size, distance_in_px, octave_values, st, pt, pt2):

    nz1 = np.logical_and(c1 != 0, np.triu(c1, 4))
    nz2 = np.logical_and(c2 != 0, np.triu(c2, 4))
    nz = np.logical_and(nz1,nz2)

    if np.sum(nz1) < 50 or np.sum(nz2)< 50:
        return []
    c1[np.tril_indices_from(c1, 4)] = 2
    c2[np.tril_indices_from(c2, 4)] = 2

    if chromosome == chromosome2:
        c1[np.triu_indices_from(c1, k=(distance_in_px+1))] = 2
        c2[np.triu_indices_from(c2, k=(distance_in_px+1))] = 2
   
    c = np.zeros(c1.shape)
    c[nz] = c1[nz] - c2[nz]
    #c = c1 - c2

    #pAll11 = np.ones_like(c[nz1]) * 2
    #pPair1 =np.ones_like(c[nz1]) * 2
    #Scales11 = np.ones_like(pAll11)
    #vAll11 = np.zeros_like(pAll11)

    #pAll22 = np.ones_like(c[nz2]) * 2
    #pPair2 =np.ones_like(c[nz2]) * 2
    #Scales22 = np.ones_like(pAll22)
    #vAll22 = np.zeros_like(pAll22)

    pAll1 = np.ones_like(c1[nz1]) * 2
    pPair1 =np.ones_like(c1[nz1]) * 2
    Scales1 = np.ones_like(pAll1)
    vAll1 = np.zeros_like(pAll1)

    pAll2 = np.ones_like(c2[nz2]) * 2
    pPair2 = np.ones_like(c2[nz2]) * 2
    Scales2 = np.ones_like(pAll2)
    vAll2  = np.zeros_like(pAll2)

    s = 10
    #curr_filter = 1
    scales = {}
    for o in octave_values:
        scales[o] = {}
        sigma = o
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gp = gaussian_filter(c, o, truncate=t, order=0)
        Gp1 = gaussian_filter(c1, o, truncate=t, order=0)
        Gp2 = gaussian_filter(c2, o, truncate=t, order=0)
        scales[o][1] = sigma

        sigma = o * 2**((2-1)/s)
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gc = gaussian_filter(c, sigma, truncate=t, order=0)
        Gc1 = gaussian_filter(c1, sigma, truncate=t, order=0)
        Gc2 = gaussian_filter(c2, sigma, truncate=t, order=0)
        scales[o][2] = sigma

        Lp = Gp - Gc
        Gp = []
        Lp1 = Gp1 - Gc1
        Gp1 = []
        Lp2 = Gp2 - Gc2
        Gp2 = []

        sigma = o * 2**((3-1)/s)
        w = 2*math.ceil(2*sigma)+1
        t = (((w - 1)/2)-0.5)/sigma
        Gn = gaussian_filter(c, sigma, truncate=t, order=0)
        Gn1 = gaussian_filter(c1, sigma, truncate=t, order=0)
        Gn2 = gaussian_filter(c2, sigma, truncate=t, order=0)
        scales[o][3] = sigma

        #Lp = Gp - Gc
        Lc = Gc - Gn
        Lc1 = Gc1 - Gn1
        Lc2 = Gc2 - Gn2

        locMaxP1 = maximum_filter(
            Lp1, footprint=np.ones((3, 3)), mode='constant')
        locMaxC1 = maximum_filter(
            Lc1, footprint=np.ones((3, 3)), mode='constant')
        locMaxP2 = maximum_filter(
            Lp2, footprint=np.ones((3, 3)), mode='constant')
        locMaxC2 = maximum_filter(
            Lc2, footprint=np.ones((3, 3)), mode='constant')

        for i in range(3, s + 2):
            #curr_filter += 1
            Gc = Gn
            Gc1 = Gn1
            Gc2 = Gn2

            sigma = o * 2**((i)/s)
            w = 2*math.ceil(2*sigma)+1
            t = ((w - 1)/2 - 0.5)/sigma
            Gn = gaussian_filter(c, sigma, truncate=t, order=0)
            Gn1 = gaussian_filter(c1, sigma, truncate=t, order=0)
            Gn2 = gaussian_filter(c2, sigma, truncate=t, order=0)
            scales[o][i+1] = sigma
            
            Ln = Gc - Gn
            Ln1 = Gc1 - Gn1
            Ln2 = Gc2 - Gn2

            dist_params1 = expon.fit(np.abs(Lc1[nz1]))
            pval1 = 1 - expon.cdf(np.abs(Lc1[nz1]), *dist_params1)
            dist_params2 = expon.fit(np.abs(Lc2[nz2]))
            pval2 = 1 - expon.cdf(np.abs(Lc2[nz2]), *dist_params2)
            params = norm.fit(Lc[nz])
            diff_pval1 = norm.cdf(Lc[nz1],
                          loc=params[0],
                          scale=params[1])
            #params = norm.fit(Lc[nz])
            diff_pval2 = norm.cdf(Lc[nz2],
                          loc=params[0],
                          scale=params[1])

            np.nan_to_num(diff_pval1, copy=False, posinf=1, neginf=1, nan=1)
            diff_pval1[diff_pval1 > 0.5] = 1 - diff_pval1[diff_pval1 > 0.5]
            diff_pval1*=2
            np.nan_to_num(diff_pval2, copy=False, posinf=1, neginf=1, nan=1)
            diff_pval2[diff_pval2 > 0.5] = 1 - diff_pval2[diff_pval2 > 0.5]
            diff_pval2*=2
            np.nan_to_num(pval1, copy=False, posinf=1, neginf=1, nan=1)
            np.nan_to_num(pval2, copy=False, posinf=1, neginf=1, nan=1)
            #pval[pval > 0.5] = 1 - pval[pval > 0.5]
            #pval *= 2

            locMaxN1 = maximum_filter(
                Ln1, footprint=np.ones((3, 3)), mode='constant')
            locMaxN2 = maximum_filter(
                Ln2, footprint=np.ones((3, 3)), mode='constant')

            willUpdate1 = np.logical_and \
                .reduce((Lc1[nz1] > vAll1, Lc1[nz1] == locMaxC1[nz1],
                         np.logical_or(Lp1[nz1] == locMaxP1[nz1],
                                       Ln1[nz1] == locMaxN1[nz1]),
                         Lc1[nz1] > locMaxP1[nz1],
                         Lc1[nz1] > locMaxN1[nz1]))
            willUpdate2 = np.logical_and \
                .reduce((Lc2[nz2] > vAll2, Lc2[nz2] == locMaxC2[nz2],
                         np.logical_or(Lp2[nz2] == locMaxP2[nz2],
                                       Ln2[nz2] == locMaxN2[nz2]),
                         Lc2[nz2] > locMaxP2[nz2],
                         Lc2[nz2] > locMaxN2[nz2]))

            vAll1[willUpdate1] = Lc1[nz1][willUpdate1]
            Scales1[willUpdate1] = scales[o][i]
            pAll1[willUpdate1] = pval1[willUpdate1]
            pPair1[willUpdate1] = diff_pval1[willUpdate1]
            Lp1 = Lc1
            Lc1 = Ln1
            locMaxP1 = locMaxC1
            locMaxC1 = locMaxN1

            vAll2[willUpdate2] = Lc2[nz2][willUpdate2]
            Scales2[willUpdate2] = scales[o][i]
            pAll2[willUpdate2] = pval2[willUpdate2]
            pPair2[willUpdate2] = diff_pval2[willUpdate2]
            Lp2 = Lc2
            Lc2 = Ln2
            locMaxP2 = locMaxC2
            locMaxC2 = locMaxN2


    pFound1 = pAll1 != 2
    pFound2 = pAll2 != 2
    if len(pFound1) < 10000 or len(pFound2) < 10000:
        return []
    _, pCorrect1, _, _ = multipletests(pAll1[pFound1], method='fdr_bh')
    _, pCorrect2, _, _ = multipletests(pAll2[pFound2], method='fdr_bh')

    pAll1[pFound1] = pCorrect1
    pAll2[pFound2] = pCorrect2

    #_, pCorrect1, _, _ = multipletests(pPair1[pFound1], method='fdr_bh')
    #_, pCorrect2, _, _ = multipletests(pPair2[pFound2], method='fdr_bh')

    #pPair1[pFound1] = pCorrect1
    #pPair2[pFound2] = pCorrect2

    #print(np.sum(pPair1<0.2),np.sum(pPair2<0.2))
    o1 = np.ones_like(c1)
    o1[nz1] = pAll1
    pair1 = np.ones_like(c1)
    pair1[nz1] = pPair1

    v1 = np.ones_like(c1)
    v1[nz1] = vAll1
    v2 = np.ones_like(c2)
    v2[nz2] = vAll2

    #x1, y1 = np.where(np.logical_and.reduce((o1 < pt, pair1 < pt2, v1>v2))) #change
    #sig_count = np.sum(o1 < pt)
    #x1, y1 = np.unravel_index(np.argsort(o1.ravel()), o1.shape)
    x1, y1 = np.where(o1<pt)
    so1 = np.ones_like(c1)
    so1[nz1] = Scales1
    #x1 = x1[:sig_count]
    #y1 = y1[:sig_count]
    xyScales1 = so1[x1, y1]
    

    o2 = np.ones_like(c2)
    o2[nz2] = pAll2
    pair2 = np.ones_like(c2)
    pair2[nz2] = pPair2
    #x2, y2 = np.where(np.logical_and.reduce((o2 < pt, pair2 < pt2, v2>v1))) #change
    #sig_count = np.sum(o2 < pt)
    #x2, y2 = np.unravel_index(np.argsort(o2.ravel()), o2.shape)
    x2, y2 = np.where(o2<pt)
    so2 = np.ones_like(c2)
    so2[nz2] = Scales2
    #x2 = x2[:sig_count]
    #y2 = y2[:sig_count]
    xyScales2 = so2[x2, y2]
    nonsparse = x1 != 0
    
    for i in range(len(xyScales1)):
        s = math.ceil(xyScales1[i])
        cc1 = np.sum(nz1[x1[i]-s:x1[i]+s+1, y1[i]-s:y1[i]+s+1]) / \
            ((2*s+1)**2)
        s = 2*s
        cc2 = np.sum(nz1[x1[i]-s:x1[i]+s+1, y1[i]-s:y1[i]+s+1]) / \
            ((2*s+1)**2)
        if cc1 < st or cc2 < 0.6:
            nonsparse[i] = False
    x1 = x1[nonsparse]
    y1 = y1[nonsparse]

    nonsparse = x2 != 0
    for i in range(len(xyScales2)):
        s = math.ceil(xyScales2[i])
        cc1 = np.sum(nz2[x2[i]-s:x2[i]+s+1, y2[i]-s:y2[i]+s+1]) / \
            ((2*s+1)**2)
        s = 2*s
        cc2 = np.sum(nz2[x2[i]-s:x2[i]+s+1, y2[i]-s:y2[i]+s+1]) / \
            ((2*s+1)**2)
        if cc1 < st or cc2 < 0.6:
            nonsparse[i] = False

    x2 = x2[nonsparse]
    y2 = y2[nonsparse]

    if len(x1) == 0 or len(x2)==0:
        return []

    def nz_mean(vals):
        return np.mean(vals[vals != 0])

    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])

    if chromosome == chromosome2:
        means = np.vectorize(diag_mean, excluded=['map'])(k=y1-x1, map=c1)
        passing_indices = c1[x1, y1] > 2*means #change
        if len(passing_indices) == 0 or np.sum(passing_indices) == 0:
            return []
        x1 = x1[passing_indices]
        y1 = y1[passing_indices]
    if chromosome == chromosome2:
        means = np.vectorize(diag_mean, excluded=['map'])(k=y2-x2, map=c2)
        passing_indices = c2[x2, y2] > 2*means #change
        if len(passing_indices) == 0 or np.sum(passing_indices) == 0:
            return []
        x2 = x2[passing_indices]
        y2 = y2[passing_indices]

    def get_labels(x,y,o):
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
        return label_matrix, num_features

    out1 = []
    out2 = []
    label_matrix1, num_features1 = get_labels(x1,y1,o1)
    label_matrix2, num_features2 = get_labels(x2,y2,o2)

    for label in range(1, num_features1+1):
        indices = np.argwhere(label_matrix1 == label)
        i = np.argmin(o1[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        out1.append([_x+start, _y+start, o1[_x, _y], so1[_x, _y]])

    for label in range(1, num_features2+1):
        indices = np.argwhere(label_matrix2 == label)
        i = np.argmin(o2[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        out2.append([_x+start, _y+start, o2[_x, _y], so2[_x, _y]])

    ################### report the differential ones only
    #print(out1, start)
    #def greater_than_NB(diff, x, y, pt2):
    #    return np.logical_and.reduce((diff[x,y]<pt2, diff[x]))
    diff_out1 = [o for o in out1 if pair1[o[0]-start,o[1]-start] < pt2 and v1[o[0]-start,o[1]-start]>v2[o[0]-start,o[1]-start]]  
    diff_out2 = [o for o in out2 if pair2[o[0]-start,o[1]-start] < pt2 and v2[o[0]-start,o[1]-start]>v1[o[0]-start,o[1]-start]]
    return out1, diff_out1, out2, diff_out2


def regulator(f1, f2, norm_method, CHRM_SIZE, outdir, bed1="", bed2="",
              res=5000,
              sigma0=1.6,
              s=10,
	      pt=0.1,
              pt2=0.1,
              st=0.88,
              octaves=2,
              verbose=True,
              nprocesses=4,
              distance_filter=2000000,
              bias1=False,
              bias2=False,
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

    if f1.endswith(".hic"):                       
        x1, y1, v1 = read_hic_file(f1, norm_method, CHRM_SIZE, distance_in_bp, chromosome,chromosome2, res)
    elif f1.endswith(".cool"):
        x1, y1, v1, res = read_cooler(f1, distance_in_bp, chromosome,chromosome2, norm_method)
    elif f1.endswith(".mcool"):
        x1, y1, v1 = read_mcooler(f1, distance_in_bp, chromosome,chromosome2, res, norm_method)
    else:
        x1, y1, v1 = read_pd(f1, distance_in_bp, bias1, chromosome, res)
   
    if f2.endswith(".hic"):                      
        x2, y2, v2 = read_hic_file(f2, norm_method, CHRM_SIZE, distance_in_bp, chromosome,chromosome2, res)
    elif f2.endswith(".cool"):
        x2, y2, v2, res2 = read_cooler(f2, distance_in_bp, chromosome,chromosome2, norm_method)
        if res2 != res:
            raise ValueError('Both contact maps should have the same resolution.')
    elif f2.endswith(".mcool"):
        x2, y2, v2 = read_mcooler(f2, distance_in_bp, chromosome,chromosome2, res, norm_method)
    else:
        x2, y2, v2 = read_pd(f2, distance_in_bp, bias2, chromosome, res)

    

    if len(v1)==0 or len(v1)==0:
        return [] 
    print("Normalizing contact map...")
    
    distance_in_px = int(math.ceil(distance_in_bp // res))
    if chromosome == chromosome2:
        n1 = max(max(x1), max(y1)) + 1
        n2 = max(max(x2), max(y2)) + 1
        n = max(n1,n2)
        
        normalize_sparse(x1, y1, v1, res, distance_in_px)
        normalize_sparse(x2, y2, v2, res, distance_in_px)

        CHUNK_SIZE = max(2*distance_in_px, 2000)
        overlap_size = distance_in_px

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

        print("Loop calling...")
        with Manager() as manager:
            o = manager.list()
            i = 0
            processes = []
            for i in range(len(start)):
                # create the currnet block
                indx1 = np.logical_and.reduce((x1 >= start[i], x1 < end[i], y1 >= start[i],y1 < end[i]))
                indx2 = np.logical_and.reduce((x2 >= start[i], x2 < end[i], y2 >= start[i],y2 < end[i]))

                xc1 = x1[indx1] - start[i]
                yc1 = y1[indx1] - start[i]
                vc1 = v1[indx1]

                xc2 = x2[indx2] - start[i]
                yc2 = y2[indx2] - start[i]
                vc2 = v2[indx2]

                cc1 = np.zeros((CHUNK_SIZE,CHUNK_SIZE))
                cc1[xc1,yc1] = vc1
                cc2 = np.zeros((CHUNK_SIZE,CHUNK_SIZE))
                cc2[xc2,yc2] = vc2
                #cc = cc1 - cc2
                #             
                p = Process(target=process_block, args=(
                    i, start, end, overlap_size, cc1, cc2, chromosome,chromosome2, res, distance_in_px, octave_values, o, st, pt, pt2))
                p.start()
                processes.append(p)
                if len(processes) >= nprocesses or i == (len(start) - 1):
                    for p in processes:
                        p.join()
                    processes = []
            return list(o)
        
    else:
        n1 = max(x) + 1
        n2 = max(y) + 1
        inter_normalize_map(v)
	
 

def process_block(i, start, end, overlap_size, cc1, cc2, chromosome,chromosome2, res, distance_in_px, octave_values, o, st, pt, pt2):
    print("Starting block ", i+1, "/", len(start), "...", sep='')
    if i == 0:
        mask_size = -1
    elif i == len(start)-1:
        mask_size = end[i-1] - start[i]
    else:
        mask_size = overlap_size
    loops1, diff_loops1, loops2, diff_loops2 = diff_mustache(
        cc1, cc2, chromosome,chromosome2, res, start[i], end[i], mask_size, distance_in_px, octave_values, st, pt, pt2)
    for loop in list(loops1):
        if loop[0] >= start[i]+mask_size or loop[1] >= start[i]+mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3], 1])
    for loop in list(diff_loops1):
        if loop[0] >= start[i]+mask_size or loop[1] >= start[i]+mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3], 2])
    for loop in list(loops2):
        if loop[0] >= start[i]+mask_size or loop[1] >= start[i]+mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3], 3])
    for loop in list(diff_loops2):
        if loop[0] >= start[i]+mask_size or loop[1] >= start[i]+mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3], 4])

    print("Block", i+1, "done.")


def main():
    start_time = time.time()
    args = parse_args(sys.argv[1:])
    print("\n")

    f1 = args.f_path1
    f2 = args.f_path2
    if args.bed1 and args.mat1:
        f1 = args.mat1
    if args.bed2 and args.mat2:
        f2 = args.mat2

    if not os.path.exists(f1) or not os.path.exists(f2):
        print("Error: Couldn't find the specified contact files")
        return
    res = parseBP(args.resolution)
    if not res:
        print("Error: Invalid resolution")
        return
    CHR_LIST_FLAG = False
    CHR_COOL_FLAG = False
    CHR_HIC_FLAG  = False
    if not args.chromosome or args.chromosome == 'n':
        if f1.endswith(".cool") or f1.endswith(".mcool"):
            CHR_COOL_FLAG = True
        elif f1.endswith(".hic"):
            CHR_HIC_FLAG = True           
        elif len(args.chromosome>1):
            print("Error: For this data type you should enter only one chromosome name.")
            return 
        else:
            print("Error: Please enter the chromosome name.")
            return
    elif len(args.chromosome) > 1:
        CHR_LIST_FLAG = True
        
    


    distFilter = parseBP(args.distFilter)#change
    if not distFilter: 
        if 200*res >= 2000000:
            distFilter = 200*res
            print("The distance limit is set to {}bp".format(200*res))
        elif 2000*res <= 2000000:
            distFilter = 2000*res
            print("The distance limit is set to {}bp".format(2000*res))
        else:
            distFilter = 2000000
            print("The distance limit is set to 2Mbp")			
    elif distFilter < 200*res:
        print("The distance limit is set to {}bp".format(200*res))
        distFilter = 200*res
    elif distFilter > 2000*res:
        print("The distance limit is set to {}bp".format(2000*res))
        distFilter = 2000*res
    elif distFilter > 2000000:
        distFilter = 2000000
        print("The distance limit is set to 2Mbp")

    #distFilter = 4000000
    chrSize_in_bp = False
    if CHR_COOL_FLAG:
        # extract all the chromosome names big enough to run mustache on
        chr_list = []
        if f1.endswith(".cool"):
            clr = cooler.Cooler(f1)
        else: #mcooler
            uri = '%s::/resolutions/%s' % (f1, res)
            clr = cooler.Cooler(uri)
        for i, chrm in enumerate(clr.chromnames):
            if clr.chromsizes[i]>1000000:
                chr_list.append(chrm)
    elif CHR_HIC_FLAG:
        hic = open(f1, 'rb')
        chrs, resolutions, masterindex, genome, metadata = read_header(hic)
        chr_list = [chrs[i][1] for i in range(1,len(chrs))]
        chrSize_in_bp = {}
        for i in range(1,len(chrs)):        
            chrSize_in_bp["chr"+str(chrs[i][1]).replace("chr",'')] = chrs[i][2] 
    else:
        chr_list = args.chromosome.copy()        

    if (args.chromosome2 and args.chromosome2 != 'n') and (len(chr_list) != len(args.chromosome2)):
        print("Error: the same number of chromosome1 and chromosome2 should be provided.")
        return
    elif type(args.chromosome2)==list:
        chr_list2 = args.chromosome2.copy()
    else:
        chr_list2 = chr_list.copy()


    CHRM_SIZE = False    
    if args.chrSize_file and (not chrSize_in_bp):
        csz_file = args.chrSize_file
        csz = pd.read_csv(csz_file,header=None,sep='\t')
        chrSize_in_bp = {}
        for i in range(csz.shape[0]):
            chrSize_in_bp["chr"+str(csz.iloc[i,0]).replace('chr','')] = csz.iloc[i,1]
           
    first_chr_to_write = True
    for i, (chromosome,chromosome2) in enumerate(zip(chr_list,chr_list2)):
        if chrSize_in_bp:
            CHRM_SIZE = chrSize_in_bp["chr"+str(chromosome).replace('chr','')]
        biasf1 = False
        if args.biasfile1:
            if os.path.exists(args.biasfile1):
                biasf = args.biasfile1
            else:
                print("Error: Couldn't find the specified bias file1")
                return
        biasf2 = False
        if args.biasfile2:
            if os.path.exists(args.biasfile2):
                biasf2 = args.biasfile2
            else:
                print("Error: Couldn't find the specified bias file2")
                return
        o = regulator(f1, f2, args.norm_method, CHRM_SIZE, args.outdir,
						  bed1=args.bed1,
                                                  bed2=args.bed2,
						  res=res,
						  sigma0=args.s_z,
						  s=args.s,
						  verbose=args.verbose,
						  pt=args.pt,
                                                  pt2=args.pt2,
						  st=args.st,
						  distance_filter=distFilter,
						  nprocesses=args.nprocesses,
						  bias1=biasf1,
                                                  bias2=biasf2,
						  chromosome=chromosome,
						  chromosome2=chromosome2,
						  octaves=args.octaves)
        if i==0:      
            with open(args.outdir+'.loop1', 'w') as out_file1:
                out_file1.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
            with open(args.outdir+'.diffloop1', 'w') as out_file2:
                out_file2.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
            with open(args.outdir+'.loop2', 'w') as out_file3:
                out_file3.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
            with open(args.outdir+'.diffloop2', 'w') as out_file4:
                out_file4.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")

        if o == []:
            print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o),chromosome,args.pt,"%.2f" % (time.time()-start_time)))
            start_time = time.time()
            continue

        #if first_chr_to_write:
        #    first_chr_to_write = False
        #print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o),chromosome,args.pt,"%.2f" % (time.time()-start_time)))
        counter1 = counter2 = counter3 = counter4 = 0
        with open(args.outdir+'.loop1', 'a') as out_file1, open(args.outdir+'.diffloop1', 'a') as out_file2, open(args.outdir+'.loop2', 'a') as out_file3, open(args.outdir+'.diffloop2', 'a') as out_file4:
            #out_file.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
            for significant in o:
                if significant[4]==1:
                    counter1+=1
                    out_file1.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
		               str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
		               '\t' + str(significant[3]) + '\n')
                elif significant[4]==2:
                    counter2+=1
                    out_file2.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
                               str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
                               '\t' + str(significant[3]) + '\n')
                elif significant[4]==3:
                    counter3+=1
                    out_file3.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
                               str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
                               '\t' + str(significant[3]) + '\n')
                elif significant[4]==4:
                    counter4+=1
                    out_file4.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
                               str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
                               '\t' + str(significant[3]) + '\n')

        print(f"({counter1},{counter3}) loops and ({counter2},{counter4}) differential-loops found in chrmosome={chromosome} for detection-fdr<{args.pt} and difference-fdr<{args.pt2} in {time.time()-start_time:.2f}sec")
        #else:
        #    print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o),chromosome,args.pt,"%.2f" % (time.time()-old_time)))
        #    with open(args.outdir, 'a') as out_file:
        #        for significant in o:
        #            out_file.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
	#	                   str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
	#		           '\t' + str(significant[3]) + '\n')
        start_time = time.time()


if __name__ == '__main__':
    main()
