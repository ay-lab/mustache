# Mustache

Mustache (Multi-scale Detection of Chromatin Loops from Hi-C and Micro-C Maps using Scale-Space Representation) is a tool by Abbas Roayaei Ardakany, Halil Tuvan Gezer, Stefano Lonardi and Ferhat Ay (ferhatay@lji.org).
Mustache is a tool for multi-scale detection of chromatin loops from Hi-C and Micro-C contact maps. Mustache uses recent technical advances in scale-space theory in Computer Vision to detect blob-shaped objects in a multi-scale representation of the contact map parametrized by the size of the smoothing kernel.
For more information read the full paper on <a href="https://www.biorxiv.org/content/10.1101/2020.02.24.963579v1">bioaRxiv</a>.

## Installation

See below for usage examples.

### Conda

Conda is the recommended way of running mustache as it will take care of the dependencies.

```bash
git clone https://github.com/ay-lab/mustache
conda env create -f ./mustache/environment.yml
conda activate mustache
python -m mustache  -f ./mustache/data/chr21_5kb.RAWobserved -b ./mustache/data/chr21_5kb.KRnorm -c 21 -r 5kb -o chr21_out5.tsv -pt 0.05
```

### Docker

We have a Docker container that allows running Mustache out of the box. You can <a href="https://docs.docker.com/storage/bind-mounts/">mount</a> the necessary input and output locations and run Mustache as follows.
```bash
docker run -it aylab/mustache
mustache -f /mustache/data/chr21_5kb.RAWobserved -b /mustache/data/chr21_5kb.KRnorm -c 21 -r 5kb -o ./chr21_out5.tsv -pt 0.05
```

### PIP

```bash
pip3 install mustache-hic
```

### Github

Make sure you have Python 3 installed, along with all the dependencies listed.

```bash
git clone https://github.com/ay-lab/mustache
cd mustache
./mustache/mustache.py ...arguments
```

### Dependencies

Mustach uses these python packages:
Check [here](environment.yml) for a list of dependancy versions that we know are working with Mustache.

1. python >= 3.6
1. numpy
1. pandas
1. matplotlib
1. seaborn
1. scipy
1. statsmodels
1. pathlib
1. cooler
1. hic-straw

## Examples

#### Example 1: Running Mustache with a contact map and a normalization/bias vector

- Run Mustache on provided example data for chromosome 21 of HMEC cell line from Rao et al. (selected due to file size restrictions) with KR normalization in 5kb resolution as follows.

```bash
mustache -f ./data/chr21_5kb.RAWobserved -b ./data/chr21_5kb.KRnorm -c 21 -r 5kb -pt 0.1 -o chr21_out.tsv
```

where -f is the raw contact map, -b is the bias (normalization vector) file, -c is the subject chromosome, -r is the resolution, and -o is the output file.

#### Example 2: Running Mustache with a .hic file

- Acquire the .hic format file for HFFc6 Micro-C from <a href="https://data.4dnucleome.org/files-processed/4DNFIPC7P27B/">4D Nucleome Data Portal</a>. Run Mustache as follows.

```bash
mustache -f ./4DNFIPC7P27B.hic -c 7 -r 5kb -pt 0.001 -o hic_out.tsv
```

where -f is our input file, -c is the subject chromosome, -r is the resolution, and -o is the output file.

#### Example 3: Running Mustache with a .cool file

```bash
wget ftp://cooler.csail.mit.edu/coolers/hg19/Rao2014-GM12878-MboI-allreps-filtered.5kb.cool
mustache -f ./Rao2014-GM12878-MboI-allreps-filtered.5kb.cool -c chr12 -r 5kb -pt 0.05 -o cooler_out.tsv
```

where -f is our input file, -c is the subject chromosome, -r is the resolution, and -o is the output file.

## Parameters

| Short                 | Long             | Meaning                                                                                                 |
| --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| _Required Parameters_ |                  |                                                                                                         |
| **-f**                | **--file**       | Location of contact map. (See below for format.)                                                        |
| **-r**                | **--resolution** | Resolution of the provided contact map.                                                                 |
| **-o**                | **--outfile**    | Name of the output file.                                                                                |
| _Optional Parameters_ |                  |                                                                                                         |
| **-b**                | **--biases**     | Location of biases (normalization) file for contact map (See below for format).                         |
| **-p**                | **--processes**  | Number of parallel processes to run. Default is 4. Increasing this will also increase the memory usage. |
| **-pt**               | **--pThreshold** | P-Value threshold for an interaction to be reported in the final output file. Default is 0.2            |
| **-sz**               | **--sigmaZero**  | Sigma0 parameter for Mustache. Default is experimentally chosen for 5Kb resolution.                     |
| **-oc**               | **--octaves**    | Octaves parameter for Mustache. Default is 2.                                                           |
| **-i**                | **--iterations** | Iteration count parameter for Mustache. Default is experimentally chosen for 5Kb resolution.            |
| **-V**                | **--version**    | Shows the version of the tool.                                                                          |

### Input Formats

Input map can be one of the following types.

#### 1. Text format (contact counts file + bias file)

Similar to Hi-C analysis tools previously developed by our lab (<a href="https://github.com/ay-lab/selfish">Selfish</a> and <a href="https://github.com/ay-lab/fithic">FitHiC</a>), we allow a simple, readable textual input format for Mustache.

To use this input mode, we require a contact map and a bias/normalization vector file.

1a. Contact map files need to have the following format. They must not have a header. The values must be separated by a tab.

| Chromosome 1 | Midpoint 1 | Chromosome 2 | Midpoint 2 | Contact Count |
| ------------ | ---------- | ------------ | ---------- | ------------- |
| chr1         | 5000       | chr1         | 65000      | 438           |
| chr1         | 5000       | chr1         | 85000      | 12            |
| ...          | ...        | ...          | ...        | ...           |

1b. Bias files need to have the following format. They must not have a header. Bias file must use the same midpoint format as the contact maps.

Bias file is a list of normalization factors. This means contact counts will be _divided_ by their corresponding factors.

| Chromosome | Midpoint | Factor |
| ---------- | -------- | ------ |
| chr1       | 5000     | NaN    |
| chr1       | 10000    | 1.12   |
| chr1       | 15000    | 0.1    |

#### 2. Juicer .hic Files

Mustache uses <a href="https://github.com/aidenlab/straw">Juicer's</a> straw tool to read .hic files.

#### 3. Cooler .cool, and .mcool Files

Mustache uses <a href="https://github.com/mirnylab/cooler">Cooler package to read .cool, and .mcool files.</a>

### Output

Output of Mustache is a TSV file and is formatted as follows

`| Bin 1 Chromosome | Bin 1 Start | Bin 1 End | Bin 2 Chromosome | Bin 2 Start | Bin 2 End | FDR | Mustache Scale for this Detection |`
