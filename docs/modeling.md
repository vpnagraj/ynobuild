# Modeling

## Overview

ynobuild classifies a failed container build into one of three failure classes from its build log. The classifier is a multi-layer perceptron (MLP) that takes TF-IDF features of the log excerpt as input. Its performance is compared against three shallow models trained on the same data.

The MLP is implemented in PyTorch. Modeling code lives in `src/ynbtriage/model/`. Training can be executed from the `ynbtriage` CLI or from the app's 'Train' screen, and the trained model is served to the app by the `model` service (`src/ynbtriage/serve.py`).

All counts and scores presented here come from `models/final/metrics.json`, which captures the parameterization and performance of the MLP as currently implemented, along with benchmark results against baseline models.

## Task and data

Each labeled build has one of three classes, defined in `taxonomy/taxonomy_map.yaml`:

| Class | Meaning |
|---|---|
| `environment_decay` | The recipe is unchanged but the ecosystem it depends on has moved: packages, URLs or parent image references are broken. |
| `recipe_error` | The Dockerfile includes mistakes, is unparseable, or refers to files that are not there. |
| `build_execution` | A command ran and failed at execution time for another reason: compiler errors, resource limits, platform mismatches, network or authentication failures. |

`ynbtriage assign-splits` divides confirmed builds into train, validation, test and gold sets, stratified by class with a fixed seed (7400). Per class, 20% go to gold, 15% to validation, 15% to test and the rest to training. Gold is never loaded by any modeling code and is reserved for a final check.

| Split | Builds | Used for |
|---|---|---|
| train | 522 | Fitting the featurizer, the baselines and the MLP's weights |
| validation | 156 | Early stopping and comparing configurations |
| test | 156 | Reporting performance |

Training set class counts are roughly balanced: 189 `environment_decay`, 161 `recipe_error` and 172 `build_execution`. Validation and test each contain 56 `environment_decay`, 48 `recipe_error` and 52 `build_execution` builds.

Note that the splits are stratified by class but not grouped by tool. Several builds of the same tool (different components, sometimes with very similar logged failures) can fall on both sides of a split: 98 of 480 tools appear in more than one split, and 82 of the 156 test builds share a tool with a training build. This likely inflates all reported scores, for the baselines as well as the MLP.

## MLP

### Training

The MLP is a feed-forward neural network with one hidden layer. The PyTorch implementation handles the key steps in the training process:

1. **Parameter initialization**: Weights are initialized as small random numbers, such that the 64 units in the hidden layer start out different from each other and can learn to detect different patterns.
2. **Forward propagation**: A group of 32 training builds, each represented as a 256-dimensional feature vector, is fed through the network. The hidden layer combines those numbers into 64 new values, and the output layer turns those into three scores, one per failure class. The highest score is the network's current guess.
3. **Loss**: The loss is a single number measuring how wrong the guesses were. The scores are converted to probabilities, and the loss is large when the network gives a low probability to a build's true class and small when it gives a high one. Note that this is averaged over the 32 builds at this step.
4. **Backpropagation**: PyTorch works backward from the loss through each layer to figure out how much each individual weight contributed to the error. It does this automatically, using the chain rule.
5. **Gradient estimation**: The result is a gradient for every weight, which is a scalar indicating which direction, up or down, would reduce the loss and how strongly. Because it is based on 32 builds rather than all 522, it is an estimate of the direction that would help on the whole training set.
6. **Gradient descent**: Every weight is nudged a small step in the direction that reduces the loss. The Adam optimizer adjusts the step size for each weight automatically, and a small penalty keeps weights from growing too large.
7. **Repeat**: Steps 2–6 are repeated for every group of 32 builds (one pass over all the training builds is an epoch), and epochs are repeated, up to 200 times. After each epoch the network's loss on the validation builds is measured. Training stops once validation loss has not improved for 25 epochs, and the weights from the best epoch are kept.

### Hyperparameters of the final model

| Setting | Value | Notes |
|---|---|---|
| Input features | `tfidf` | TF-IDF reduced to 256 dimensions |
| Hidden layers | `[64]` | One hidden layer, 64 units |
| Activation | ReLU | |
| Dropout | 0.2 | Hidden layer only |
| Optimizer | Adam | |
| Learning rate | 0.001 | |
| Weight decay (L2) | 0.0001 | |
| Batch size | 32 | Number of build logs in each group used for one weight update |
| Max epochs | 200 | |
| Early-stopping patience | 25 epochs | On validation loss |
| Class-weighted loss | Yes | Inverse class frequency |
| Seed | 7400 | |

### Input representation

Build logs are long and the cause of a failure is usually near the first error line or at the end. Each log is reduced to an excerpt before any model sees it (`excerpt.py`):

1. Find the first line matching a list of error patterns (`error:`, `E: `, `fatal:`, `Killed`, `exec format error`, `curl: (N)` and others).
2. Keep 5 lines before it and 15 after, plus the last 30 lines of the log.
3. Replace values that differ between builds but carry no failure signal: long hex strings (`<HEX>`), timestamps (`<TS>`) and byte sizes (`<SIZE>`).
4. Truncate to the last 4,000 characters.

Every model, baselines included, uses the same excerpts.

The MLP needs a fixed-length numeric vector. The final model uses TF-IDF features (`features: tfidf`, implemented in `features.py`):

1. TF-IDF over word unigrams and bigrams and over character 3- to 5-grams within words (at most 50,000 character n-grams; terms must appear in at least two training logs). The word tokenizer keeps tokens such as `libhts-dev` and `setup.py` whole.
2. Truncated SVD reduces the combined sparse matrix to 256 dense dimensions.
3. Each dimension is standardized to zero mean and unit variance.

The fitted featurizer (TF-IDF vocabulary, SVD projection and scaling) is saved as `featurizer.joblib`. The network's weights are saved separately in `model.pt`. The network's first layer expects exactly these 256 inputs, so the weights cannot be used without the featurizer.

Note that while the current final model uses the 256-dimensional TF-IDF feature vector, the training procedure can accommodate two other input options: `embed` (384-dimensional sentence embeddings from the pretrained `BAAI/bge-small-en-v1.5` encoder, used frozen) and `both` (the TF-IDF feature vector and the sentence embeddings combined).

## Benchmarks

### Baseline models

The MLP is compared against three shallow models (`baselines.py`), all of which use the same excerpts and the same splits, are fit on the training split only, and are scored on validation and test. They are implemented with scikit-learn and trained automatically alongside the MLP by `ynbtriage train`. The `ynbtriage baselines --out <run>` CLI scores them for an existing run without retraining it.

- **Majority class**: Always predicts the most common training class (`environment_decay`). Included as the floor for performance.
- **Naive Bayes on word counts**: Counts single words in each excerpt. Words must appear in at least two training logs and are not weighted with TF-IDF. The model uses multinomial Naive Bayes with add-one smoothing and picks the class under which the words in a log are most probable. Included as a simple text classifier approach.
- **TF-IDF with logistic regression**: The same word and character n-gram TF-IDF as the MLP's input, but kept at full size (no SVD), fed to multinomial logistic regression with inverse-regularization strength C = 4 and class-balanced sample weights. This is a linear model over tens of thousands of n-gram features, and is equivalent to a network with no hidden layer. Included as the strongest baseline.

### Results

Performance for MLP and baselines was measured with accuracy and macro-F1, which is the unweighted average of the F1 (harmonic mean of precision and recall for a given class) across all three classes.

| Model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| Majority class | 0.359 | 0.176 | 0.359 | 0.176 |
| Naive Bayes (word counts) | 0.808 | 0.811 | 0.891 | 0.892 |
| TF-IDF + logistic regression | 0.865 | 0.870 | 0.917 | 0.918 |
| **MLP** | 0.846 | 0.852 | **0.923** | **0.924** |

The MLP clearly outperforms the majority class and Naive Bayes baselines. Its lead over TF-IDF with logistic regression is one build on the test split (144 vs. 143 of 156 correct), and on validation the order reverses (132 vs. 135), so the two perform about the same.