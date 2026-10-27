# experiments

This directory contains notebooks and helper modules for modeling experiments. The `ynobuild` tool itself (app, API, CLI, and the served model) are maintained separately in the repo. The experiments reuse `ynobuild` components and modeling artifiacts (data loading, splits, featurizers, the training loop, metrics, and baselines) so results are directly comparable to the model in `models/final/`.

Currently there is one experiment investigating a two-stream MLP with a learned gate.

| File | Purpose |
|---|---|
| `01_two_stream_gate.ipynb` | Runs the experiment end to end by training the original MLP and the two-stream model, comparing them, and measuring what the gate learned |
| `two_stream.py` | Includes functions for the new approach (excerpt splitting, the two-stream model, the multi-seed runner, and the evaluation metrics) |

---

## Two-stream MLP with a learned gate

### Data

`ynobuild` uses container build log text to classify build failure as one of three classes: `environment_decay`, `recipe_error`, or `build_exection`. For more on this taxonomy refer to [`taxonomy/taxonomy_map.yaml`](../taxonomy/taxonomy_map.yaml).

The data are roughly 1,000 failed builds of bioinformatics and scientific computing tools, each with a build log and a human-annotated failure class. Labeled builds are split (stratified by class, fixed seed) into train (~50%), validation (~15%), test (~15%), and a gold (~20%). The database isn't included in this repository, however the tool includes a method to generate synthetic data to demonstration operation.

### Baseline

The baseline is the original `ynobuild` MLP (see [`docs/modeling.md`](../docs/modeling.md) for the full description). Each log is trimmed to an excerpt with two parts:

- **Error window:** the first line matching a list of error patterns (e.g., `error:`, `Killed`), plus 5 lines before it and 15 after.
- **Log tail:** the last 30 lines.

The two parts are concatenated into one block of text, and strings that change from build to build without saying anything about the failure (e.g., hashes, timestamps, file sizes) are masked. The text is turned into TF-IDF features over words, word pairs, and short character sequences. That gives tens of thousands of features, which SVD compresses to 256 dimensions. Those go into an MLP with one hidden layer of 64 units, which outputs a score for each class.

The original MLP already does well, with a test macro F1 above 0.9 per the original analysis. With that said, it was essentially tied with logistic regression on the full TF-IDF features. Logistic regression is a neural network with no hidden layers, so a hidden layer wasn't adding much. That casts doubt on the value of adding  more hidden layers. Instead, it might be more worthwhile to focus on designing modifications for the input representation.

### Design decisions

Merging the error window and the log tail into one block means the model can't tell which part of the log a given word came from. This experiment is based on the hypothesis that the two parts carry complementary but unequal signal, and that the error window would be the more informative of the two.

To test that, the two-stream model keeps each part of the log excerpt separate:

```
error window → TF-IDF/SVD (128-d) → [branch: hidden layer, 32 units] ─┐
                                                                       ├─ gate → weighted sum → final layer → 3 class scores
log tail     → TF-IDF/SVD (128-d) → [branch: hidden layer, 32 units] ─┘
```

The design is based on:

- **Separate features**: Each log excerpt component gets its own TF-IDF and SVD, reduced to 128 dimensions each, so the total input is still 256 dimensions like the original.
- **Branches**: Each log excerpt component goes through its own hidden layer, which produces a 32-number summary. Each branch has one hidden layer (same as the original MLP), so the comparison isn't about depth. The two-stream approach ends up with about half as many parameters as the original (8,485 vs. 16,643).
- **Gate**: The gate is a single linear layer followed by a softmax, shared across all logs. For each log, it looks at both branch summaries and outputs two weights that sum to 1. The weighted sum goes to the final layer. The gate starts at 50/50 for every log and is trained with backpropagation along with the rest of the network.

### Implementation

The code for the experiment was set up to reuse as much of the existing `ynobuild` codebase as possible. Functionality for data loading, splits, labels, the TF-IDF featurizer, the training loop, and metrics all come from `ynbtriage`. The only chnage to the original tool was to add an option ("model_fn") so a different network can be trained with the same loop. The two-stream approach is implemented in `experiments/two_stream.py`, which is loaded into the notebook as a module.

To run the experiment, start at the repo root and install `ynbtriage`plus Jupyter to run the notebook:

```bash
pip install -e ".[model]" jupyter
```

Assign `YNB_DB_PATH` to an absolute path for an annotated `ynobuild` database and make it visible to the notebook session by either exporting or including in an `.env` file:

```
YNB_DB_PATH=/absolute/path/to/ynobuild.db
```

**NOTE**: Even without genuine log data, `ynbtriage` includes a feature to simulate a DB for exploration:

```bash
ynbtriage synth-db --db /tmp/ynb-synth.db --n 600
```

Then start Jupyter from this directory, so the notebook can import `two_stream.py`, and run all cells:

```bash
jupyter lab
```

Configuration options are set in the first cell of the notebook:

| Setting | Default | Description |
|---|---|---|
| `SEEDS` | 5 seeds (7400–7404) | Number of times each model is trained, with more seeds providing a better estimate of variation between runs|
| `ANALYSIS_SPLIT` | `"val"` | Which split the gate and knockout analysis use; useful toggle when switching from `"val"` to `"test"` when experimentation is complete |
| `CFG` | `TrainConfig()` | Training settings, the same as the original model (Adam optimizer, learning rate 1e-3, weight decay 1e-4, dropout 0.2, batch size 32, early stopping after 25 epochs without improvement) |

### Evaluation

The original MLP was retrained in the same notebook, next to the two-stream model, with the same settings, splits, and seeds. Each model was trained five times, and results are reported as the mean and SD across runs. Early stopping and exploratory analysis used the validation split. Final numbers come from the test split, which wasn't used for any design decision.

The following metrics can help illuminate which part of the log the two-stream model relies on more:

1. **Gate weight:** The average weight the gate gives each log excerpt component, stratified by failure class.
2. **Contribution share:** How much of the combined branch output, by size, actually comes from each combined. It is possible that large output could offset a small gate weight, which is why this metric could be informative alongside the gate weight alone.  
3. **Knockout:** Accuracy and "flip rate" (share of logs whose prediction changes) measured by removing one branch from the trained model (without retraining) and re-scoring. 

### Results

The two-stream model performed about the same as the original MLP. The table below reports the accuracy (mean ± SD across replicates via seeds):

| Split | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Validation | Original MLP | 0.851 ± 0.003 | 0.857 ± 0.003 |
| Validation | Two-stream | 0.858 ± 0.007 | 0.863 ± 0.007 |
| Test | Original MLP | 0.924 ± 0.005 | 0.925 ± 0.006 |
| Test | Two-stream | 0.924 ± 0.007 | 0.926 ± 0.007 |

On test, the F1 per-class shifted only by 0.01 or less (original → two-stream): `environment_decay` 0.927 → 0.928, `recipe_error` 0.945 → 0.955, `build_execution` 0.903 → 0.894.

For the analysis of the contribution of the error window vs log tail, the results show that the gate put less than half its weight on the error window. Contribution shares lined up closely with the gate weights, and the knockout analysis (summarized with mean accuracy and flip prate in the table below) communicates the same message:

| Variant (test) | Accuracy | Flip rate |
|---|---|---|
| Full model | 0.924 | 0 |
| Without error window | 0.918 | 0.041 |
| Without log tail | 0.860 | 0.114 |

In other words, removing the tail from the log excerpt was much more costly to performance than removing the error window. This (along with the findings about gate weight and contribution shares) suggest that the log tail harbors more of the build failure signal than the first-matched error pattern in the log. 

It's also worth noting that model accuracy across the board is higher in test compared to validation. This finding aligns with what we saw in the original MLP analysis. While it seems a little counterintuitive (i.e., we might expect the validation to be *more* optimistic), the difference likely reflects underlying challenges in splitting the log data in the first place. The test set may be composed of "easier" cases. So while the discrepancy doesn't say anything about the model performance (and is internally consistent for the comparisons that are the subject of this experiment), it does suggest that the splits (and more importantly, the overall amount of labeled data) should be revisited in the future.

### Lessons learned

1. The original hypothesis that the error window is more informative than log tail was rejected. Instead, the log tail seems to carry most of the failure signal. 
2. Design choices can add value even if it doesn't target accuracy improvement. We went into the two-stream experiment knowing that the original MLP was already performing pretty well (macro F1 > 0.9 in test set). Given the size of the dataset, there were only a handful of logs that were classified incorrectly. So even if we could achieve an accuracy bump, it would be probably only be a marginal benefit. The better use of design here was to focus on trying to explore mechanisms working inside the modeling architecture.
3. The original `ynobuild` MLP isn't suited to all deep learning architectural components. The MLP is a fully connected feedforward network, not a CNN. It takes a fixed-length vector of TF-IDF/SVD features, which describe what a log contains but not the order things appear in. Kernel sizes, strides, and pooling all depend on that order, so they aren't the levers to pull here.
4. The next step is more data. With only 156 logs per split, which logs land where matters more than the random seed. Variation of accuracy between runs was under a 1%, but validation and test differed by ~7%. More labeled logs from more tools would make comparisons like this one more reliable. A bigger dataset might also make other design choices, like more depth, worth trying without overfitting.