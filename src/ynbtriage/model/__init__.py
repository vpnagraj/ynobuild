"""Modeling for ynobuild: predict a build's failure category from its log.

Pipeline (each step is its own module so it can be read on its own):

    excerpt.py   log_tail -> error-relevant excerpt (same text for every model)
    data.py      DB -> DataFrame of (build_id, tool_name, text, label, split)
    features.py  text -> fixed-length vector (TF-IDF+SVD, sentence embedding, or both)
    mlp.py       the neural network (feed-forward MLP, >=1 hidden layer)
    train.py     the training loop: forward -> loss -> backward -> update
    baselines.py shallow benchmarks (majority class, TF-IDF + logistic regression,
                 linear probe on the MLP's own features)
    metrics.py   accuracy, macro-F1, confusion matrix
    pipeline.py  glue: run everything, write an artifact directory
    predict.py   load an artifact directory and classify new log text

    backprop.py  standalone walkthrough: one hidden layer, gradients by hand,
                 checked against PyTorch autograd. Not used by the pipeline.
    synth.py     synthetic labelled DB for development before annotation is done.

Heavy dependencies (torch, scikit-learn, sentence-transformers) are an optional
extra: `pip install ".[model]"`. Nothing in the core CLI/API imports this package
at module load time.
"""
