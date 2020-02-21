# Mustache

Mustache(Multi-scale Detection of Chromatin Loops from Hi-C and Micro-C Maps using Scale-Space Representation) is a tool by Abbas Roayaei Ardakany, Ferhat Ay (ferhatay@lji.org), Stefano Lonardi, and Halil Tuvan Gezer.
Mustache is a tool for multi-scale detection of chromatin loops from Hi-C and Micro-C contact maps.Mustache uses recent technical advances in scale-space theory in Computer Vision to detect blob-shaped objects in a multi-scale representation of the contact map parametrized by the size of the smoothing kernel. For more information read the full
paper: <a href="">TBA</a>.

## Installation

See below for usage examples.

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

Mustach uses some python packages to accomplish its mission. These are the packages used by mustache:

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. scipy
6. statsmodels
7. pathlib
8. cooler
9. hic-straw

## Examples

#### Example for running Mustache with a .hic file.

- Acquire a hic file. Here we are using <a href="https://data.4dnucleome.org/files-processed/4DNFIPC7P27B/">MicroC data from HFFc6 cells</a>
  Run Mustache as follows.

```bash
mustache -f ./4DNFIPC7P27B.hic -c 7 -r 5kb -o hic_out.tsv
```

Where -f is our input file, -c is the subject chromosome, -r is the resolution, and -o is the output file.

#### Example for running Mustache with a .cool file.

```bash
wget ftp://cooler.csail.mit.edu/coolers/hg19/Rao2014-GM12878-MboI-allreps-filtered.5kb.cool
mustache -f ./Rao2014-GM12878-MboI-allreps-filtered.5kb.cool -c chr12 -r 5kb -o cooler_out.tsv
```

Where -f is our input file, -c is the subject chromosome, -r is the resolution, and -o is the output file.

## Parameters

| Short                 | Long             | Meaning                                                                                                 |
| --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| _Required Parameters_ |                  |                                                                                                         |
| **-f**                | **--file**       | Location of contact map. (See below for format.)                                                        |
| **-r**                | **--resolution** | Resolution of the provided contact map.                                                                 |
| **-o**                | **--outfile**    | Name of the output file.                                                                                |
| _Optional Parameters_ |                  |                                                                                                         |
| **-b**                | **--biases**     | Location of biases file for contact map. (See below for format.)                                        |
| **-p**                | **--processes**  | Number of parallel processes to run. Default is 4. Increasing this will also increase the memory usage. |
| **-sz**               | **--sigmaZero**  | Sigma0 parameter for Mustache. Default is experimentally chosen for 5Kb resolution.                     |
| **-oc**               | **--octaves**    | Octaves parameter for Mustache. Default is 2.                                                           |
| **-i**                | **--iterations** | Iteration count parameter for Mustache. Default is experimentally chosen for 5Kb resolution.            |
| **-V**                | **--version**    | Shows the version of the tool.                                                                          |

### Input Formats

Input map can be one of the following types.

#### .hic Files

Mustache uses <a href="https://github.com/aidenlab/straw">Juicer's</a> straw tool to read .hic files.

#### .cooler, and .mcooler Files

Mustache uses <a href="https://github.com/mirnylab/cooler">Cooler package to read .cool, and .mcool files.</a>

#### Text Contact Maps

Contact maps need to have the following format. They must not have a header.
Values must be separated by a tab.

| Midpoint 1 | Midpoint 2 | Contact Count |
| ---------- | ---------- | ------------- |
| 5000       | 65000      | 438           |
| 5000       | 85000      | 12            |
| ...        | ...        | ...           |

#### Bias File

Bias file need to have the following format.
Bias file must use the same midpoint format as the contact maps.
Bias file must not have a header. Bias file is a list of normalization factors. This means contacts will be _divided_ by their corresponding factors.

| Factor |
| ------ |
| NaN    |
| 1.12   |
| 0.1    |

Where the line number is equal to the bin number. So if your resolution is 5Kb, first line is the factor for 0-5Kb bin.

### Output

Output of Mustache is a TSV file. It's format is as follows

`| Bin 1 Chromosome | Bin 1 Start | Bin 1 End | Bin 2 Chromosome | Bin 2 Start | Bin 2 End | FDR | Mustache Scale for this Detection |`
