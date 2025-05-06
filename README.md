# Causal Discovery Pipeline

A lightweight Python interface around Tetrad that lets you run causal‐discovery algorithms from the command line.

---

## 1  Environment setup

| Requirement | Version | Notes                                                           |
| ----------- | ------- | --------------------------------------------------------------- |
| Python      | ≥ 3.10  | Use a virtual‑env or Conda environment.                         |
| Java JDK    | ≥ 11    | The JVM is started via **JPype**; you must set `JAVA_HOME`. |

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Python deps
pip install -r requirements.txt

# Tell JPype where to find the JDK
export JAVA_HOME=/path/to/jdk
export PATH="$JAVA_HOME/bin:$PATH"
```


## 2  Configuration

All runtime options live in the YAML files under **`configs/`**.

```yaml
algorithm_name: boss # fges | pc | grasp | boss | dagma | directlingam ...
score_name: conditional_gaussian_score # see src/default_params.py
score_params:
  penalty_discount: 2
bootstrap_strategy: bootstrap100  # jackknife90, (see defaults)
num_threads: 16
```

Optional inputs:

* **Knowledge file** (`--knowledge`) – Tetrad tier / required / forbidden edge file.
* **Metadata JSON** (`--metadata`) – column dtypes etc.
* **Bootstrap presets, scores, tests** – see `src/default_params.py`.


## 3  Running the pipeline

```bash
python main.py \
  --config   configs/boss.yaml \
  --data     data/my_dataset.csv \
  --output   output/graph.txt \
  --knowledge data/knowledge.txt \
  --metadata  data/metadata.json
```



## 4  How it works

1. **`main.py`** parses CLI
2. **`src/causal_discovery.py`** Configures and runs Causal Discovery:

   * builds a `CausalDiscovery`,
   * pulls default hyper‑parameters from **`src/default_params.py`**,
   * hands control to **`TetradSearch`** (JPype wrapper).
3. **JPype** starts a JVM using `JAVA_HOME` and loads **`tetrad-current.jar`** (bundled in *src/pytetrad/resources/*).
4. The selected Tetrad algorithm (FGES, PC, BOSS, etc.) fits to the pandas DataFrame.
5. Results are written to `--output` and logged.
