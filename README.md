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
python -m venv .venv
source .venv/bin/activate

# Python deps
pip install -r requirements.txt
```

If you don't have the JDK, use SDKMAN to install it easily:
```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 
sdk default java 17.0.2-open #or your version
echo $JAVA_HOME
```
If `$JAVA_HOME` is blank, try `sdk use java 21.0.7-tem` (use your version)


## 2  Configuration

All runtime options live in the YAML files under **`configs/`**. To eliminate bootstrapping, just set `numberResampling` to 0. 

Verify the valid `algorithm_names`, `test_names` and `score_names` in `causal_discovery.py`. Feel free to add any new methods in `TetradSearch.py`.

```yaml
algorithm_name: run_pc
algorithm_params:
    conflict_rule: 1
    depth: -1
    stable_fas: true
    guarantee_cpdag: false
test_name: use_degenerate_gaussian_test
test_params:
    alpha: 0.01
    use_for_mc: False
    singularity_lambda: 0.0
bootstrap_params:
    numberResampling: 10
    percent_resample_size: 100
    add_original: true
    with_replacement: true
    resampling_ensemble: 3
    seed: -1
num_threads: 16
```

Optional inputs:

* **Knowledge file** (`--knowledge`) – Tetrad tier / required / forbidden edge file.
* **Metadata JSON** (`--metadata`) – column dtypes etc.
* **Bootstrap presets, scores, tests** – see `src/default_params.py`.


## 3  Running the pipeline
```bash
source scripts/run_example.sh

```
which runs a test for boss and direct_lingam. Here is an example call:
```bash
python3 main.py \
    --config  "configs/boss.yaml" \
    --data data/example_mixed/Xy_train.csv \
    --output "output/example_output/boss/output.txt" \
    --knowledge data/example_mixed/knowledge.txt \
    --metadata data/example_mixed/metadata.json
```



## 4  How it works

1. **`main.py`** parses CLI
2. **`src/causal_discovery.py`** Configures and runs Causal Discovery:

   * builds a `CausalDiscovery`,
   * hands control to **`TetradSearch`** (JPype wrapper).
3. **JPype** starts a JVM using `JAVA_HOME` and loads **`tetrad-current.jar`** (bundled in *src/pytetrad/resources/*).
4. The selected Tetrad algorithm (FGES, PC, BOSS, etc.) fits to the pandas DataFrame.
5. Results are written to `--output` and logged.
