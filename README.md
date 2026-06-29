# Stage_2A
This project is what I am working on during my internship at CVUT in Prague.

## Quick start

A `Makefile` sets up everything from scratch (virtual environment + dependencies).

| Command | Action |
|---------|--------|
| `make` or `make install` | Create the `venv` and install all dependencies |
| `make run` | Run `Python/Exemples.py` |
| `make data` | Regenerate all intermediate CSVs (classification + stats) |
| `make classify` | Regenerate the user classification CSVs only |
| `make stats` | Regenerate the entrance/exit stats CSVs only |
| `make clean` | Remove the venv and Python caches |
| `make help` | Show the available targets |

> The intermediate CSVs in `results/intermediate_result/` are precomputed and
> read by the scripts. They are **not** regenerated on every run (that would
> reprocess the whole dataset). Run `make data` only when the source data in
> `Database/no_duplicate/` changes.

### First time

```bash
make            # creates ./venv and installs dependencies
```

Then, to run a script:

```bash
make run                          # runs Python/Exemples.py

# or manually:
source venv/bin/activate          # activate the environment
python Python/Exemples.py         # run any script
```

Dependencies are listed in `requirements.txt` (`geopy`, `matplotlib`, `numpy`,
`pandas`, `scikit-learn`, `tqdm`).
