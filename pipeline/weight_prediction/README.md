# Counting experiments

## Counting `WeightPredictor`s

 For every combination of

    prediction_mem {clone, linear}
    optimization algorithm (e.g {PYTORCH_SGD, TENSORFLOW_SGD, ADAM, WADAM, ...})

We have a WeightPredictor

## Counting `FixFunction`s

for every combination of

    optimization algorithm (e.g {PYTORCH_SGD, TENSORFLOW_SGD, ADAM, WADAM, ...})

we have several ways for predicting

    {ms_nag, just_multiply}

## Calculation with numbers

    opt = 3 (sgd1, sgd2, wadam)
    pred_mem = 2
    pred_alg = 2

### Total prediction runs (12)

    pred_runs: 2*3*2 = 12

### Total to comapre (12)

    # Stale weights (3)
    # weight stashing (3)
    # Fully sync (3)
    # GPipe/DP (3)

Total of 24 runs per net/dataset/....

## Then we vary

    pipe length...
    flush rate...
    ...
