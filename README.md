# TomoML

## Installation

### Linux
1. Install pip packages:
```
cd TomoML
pip install -r requirements.txt
```

2. Install custom dival library:
```
cd ..
git clone https://github.com/kam45x/dival.git
cd dival
pip install -e .
```

### Windows
1. Create the conda environment:
```
cd TomoML
conda env create -f environment.yml
conda activate tomoml
```

2. Install custom dival library:
```
cd ..
git clone https://github.com/kam45x/dival.git
cd dival
pip install -e .
```