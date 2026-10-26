## Overview

The `ynobuild` model passes featurized container build failure log text data into a multi-layer perceptron (MLP). Each log is trimmed to an excerpt: 

1. **Error window**: The first line that matches a list of error patterns (e.g., `error:`, `Killed`), padded by 5 lines before and 15 lines after.
2. **Log tail**: The last 30 lines of the log file. 

The two parts are combined, and component strings that are expected to carry no failure signal (e.g., hashes, timestamps) are obscured. The excerpt is then featurized as TF-IDF, which scores how distinctive each word, word pair, and short character sequence is for a given log. With tens of thousands of features generated, the TF-IDF feature vector is reduced to 256-d that capture the main patterns of variation via SVD. Those go into an MLP with one hidden layer (64 units), which outputs a score for each of the three failure classes: `environment_decay`, `recipe_error`, and `build_execution`. The original MLP classifies build failures relatively well (macro F1 > 0.9 in test set). To improve the MLP, I initially considered adding depth. It is tempting to simply add more hidden layers, especially given precedents like ResNet. But the original results suggest that may not help. The MLP was neck-and-neck with logistic regression on the full TF-IDF features. Logistic regression acts like a neural network without hidden layers. If the logistic regression was as accurate as the MLP, should we expect that more hidden layers would help? I didn't think so, and I ultimately decided that depth probably is not useful for now. Instead, I decided to focus on the input data representation. My intuition was that concatenating the error window and log tail muddles individual contribution of these components. 

**Hypothesis**: The log excerpt components convey complementary but not equal signals, with the error window being more informative than the log tail.

## Design

To test the hypothesis, I used a "two-stream" MLP with **branches** and a **gate**: 

    error window → TF-IDF/SVD (128-d) → [branch] ─┐
                                              ├─ gate → weighted sum → class scores
    log tail     → TF-IDF/SVD (128-d) → [branch] ─┘


The log failure excerpt error window and log tail are featurized separately with TF-IDF. I've reduced each to 128-d using SVD, so the total input remains 256-d like the original. The split inputs are passed into separate *branches* that both have a single hidden layer, which differentiates learning for the error window and the log tail. While this doesn't allow us to explore extra depth or gated residuals, we can more clearly assess impact of the separate branches. A *gate* weights the two branches before combining them. The gate is learned and shared across all logs, but it computes a separate pair of weights for every log by looking at both branch summaries and outputting two weights summing to 1. The weighted sum of the two branches is the input to a final layer that produces the three class scores. The gate starts as a 50/50 split of weighting the log excerpt components, but learns through backpropagation.

## Evaluation

For this experiment, I retrained the original MLP alongside the two-stream model. Both approaches used identical configurations, train/validation/test splits, and random seeds. I ran each method five times to generate a distribution of accuracy metrics. Overall accuracy esults are based on the n=156 test split. To evaluate the separate contributions of error window and log tail, I computed *gate weight* (the average weight the gate allocates to each component), *contribution share* (the share of combined branch output by magntidue that comes from each component), and *knockout* (the impact of removing a give branch from the trained model without retraining; reported as the accuracy and the flip rate for log prediction changes).

## Results

The two-stream MLP performed comparably to the original. The per class F1 differed by 0.01 or less for all failure categories (reported here as original → two-stream): `environment_decay` 0.927 → 0.928; `recipe_error` 0.945 → 0.955; `build_execution` 0.903 → 0.894. For every class, the gate put less than half its weight on the error window, and the contribution shares closely matched the gate weights in terms of favoring log tail. The knockout results reinforced that the log tail was more informative. The table below shows the mean accuracy and flip rate of model variants.

| Model  | Accuracy | Flip rate |
|---|---|---|
| Full model | 0.924 | 0 |
| Without error window | 0.918  | 0.041 |
| Without log tail | 0.860 | 0.114  |

## Interpretation

The two-stream design didn't meaningfully improve accuracy of the MLP. However, it did provide a valuable insight: the log tail carries the bulk of the failure signal. Interestingly, this counters my experimental hypothesis. I had expected the error window to be more impactful, but in fact removing that component degraded performance *less* than removing the log tail. This suggests that a) the "first match" error window approach could be improved or b) the log tail could be extended and error window pattern match removed altogether. In terms of overall accuracy, rigor, and validity of the MLP, my instinct is that the most important next step is continuing to expand the scope and scale of labeled log failure data. More logs from more tools may enable other design decisions (e.g., depth without overfitting).