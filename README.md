# Codeine
Mapping **code** locations to **in**put **e**lements.

This repository contains the code that is required to use [tribble] and [alhazen] 
to establish and evaluate associations between the grammar coverage of input files
and the methods that are executed in a program under test.

[tribble]: https://github.com/havrikov/tribble
[alhazen]: https://dl.acm.org/doi/pdf/10.1145/3368089.3409687

## Requirements
- Java 11+
- Python 3.8+
- [pipenv](https://pipenv.pypa.io/en/latest/)

## Setup
Clone the project repository with submodules:
```bash
git clone --recurse-submodules https://github.com/havrikov/codeine.git
```

After cloning the repository, install the necessary python packages into a temporary environment:
```bash
cd codeine
pipenv install
```

Set the `ram` value in `luigi.cfg` to the amount available (in gigabytes).

## Running
1. Activate the temporary environment:
```bash
pipenv shell
```
2. Start the luigi daemon
```bash
luigid --background --port 8089
```
(Alternatively, use `./start_luigi_daemon.sh`)
3. Run the experiments
```bash
./experiments.py --random-seed 42  --workers `nproc` 2>&1 | tee -a codeine.log
```
