# Mustache

Mustache(Multi-scale Detection of Chromatin Loops from Hi-C and Micro-C Maps using Scale-Space Representation) is a tool by Abbas Roayaei Ardakany, Ferhat Ay (ferhatay@lji.org), Stefano Lonardi, and Halil Tuvan Gezer.
Mustache is a tool for multi-scale detection of chromatin loops from Hi-C and Micro-C contact maps.Mustache uses recent technical advances in scale-space theory in Computer Vision to detect blob-shaped objects in a multi-scale representation of the contact map parametrized by the size of the smoothing kernel. For more information read the full
paper: <a href="https://academic.oup.com/bioinformatics/article/35/14/i145/5529135">\*\*Selfish: discovery of differential chromatin interactions via a self-similarity measure\*\*</a>.

## Installation and usage

### PIP

```bash
pip3 install mustache-hic
mustache -f /path/to/contact/map.txt \
         -r 100kb -o ./output.tsv
```

### Github

Make sure you have Python 3 installed, along with all the dependencies listed.

```bash
git clone https://github.com/ay-lab/mustache
mustache -f /path/to/contact/map.txt \
         -r 100kb -o ./output.tsv
```

### Dependencies

Selfish uses some python packages to accomplish its mission. These are the packages used by selfish:

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. scipy
6. statsmodels
7. pathlib

## Parameters

| Short                 | Long                 | Meaning                                                                                            |
| --------------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| _Required Parameters_ |                      |                                                                                                    |
| **-f**                | **--file**           | Location of contact map. (See below for format.)                                                   |
| **-r**                | **--resolution**     | Resolution of the provided contact map.                                                            |
| **-o**                | **--outfile**        | Name of the output file.                                                                           |
| _Optional Parameters_ |                      |                                                                                                    |
| **-b**                | **--biases**         | Location of biases file for contact map. (See below for format.)                                   |
| **-sz**               | **--sigmaZero**      | Sigma0 parameter for Mustache. Default is experimentally chosen for 5Kb resolution.                |
| **-oc**               | **--octaves**        | Octaves parameter for Mustache. Default is 2.                                                      |
| **-d**                | **--distanceFilter** | Distance filter parameter for Mustache. Loops are looked for within this distance. Default is 2Mb. |
| **-i**                | **--iterations**     | Iteration count parameter for Mustache. Default is experimentally chosen for 5Kb resolution.       |
| **-V**                | **--version**        | Shows the version of the tool.                                                                     |

### Input Formats

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
