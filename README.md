# AVATAR
AdVersarial system vAriant AppRoximation

## How To
This section explains how to use this project, in particular:
Playing out Variants from a System

### Systems
#### System Variant Playout
To playout a unique system variant log from e.g. *PA System 11 3*, simply run the following command from the base directory. The script will create the variant log of the system, a train variant log, and a test variant log as txt files. Moreover, the train variants are used to create a CSV and an XES based event log suitable for process discovery.
The script also print the number of unique events, number of train, test, and system variants, and the maximum system variant length.
```python
python -m systems.playout.pa_system_11_3
```

All available system scripts can be found in [systems/playout](systems/playout).

### Processes
#### Process Variant Playout
This script plays out variants from a given Petri net. --traces refers to the number of variants to be generated. Default value is one million. The variants will be stored in *data/variants/*.
As soon as the variants are generated, the script evaluates the obtained ones against the ground truth system. If one wants to evaluate only, without playing out variants, set --eval_only to True.

```python
python -m processes.playout --system <...> --pn <....pnml> --eval_only <True/False> --traces 1000000
```

#### Process Conformance Checking
Checking the conformance and obtaining the two best Petri net models per system and process discovery algorithm, as explained in the paper (Section V.C.).
The system parameter describes the underlying system, miner describes the process discovery algorithm used. The Petri nets must be stored as PNML files located in the folder *data/pns/\<system\>*.
```python
python -m processes.conformancechecker --system <...> --miner <splitminer/fodina>
```

### AVATAR
#### Process Evaluation
The subsequent script evaluates the variants sampled from a Petri net against the ground truth system variants when --eval_only is set to *True*.

```python
python -m processes.playout --system <...> --pn <....pnml> --eval_only True
```

### SGAN Training

### SGAN Sampling

#### SGAN Evaluation


