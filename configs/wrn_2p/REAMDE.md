# THIS IS DEPRECATED
# Some explanation

* `cifar10` and `cifar100` dirs are for wrn16x4,
* `cifar100_wrn28x10` is for wrn28x10.
* (will change names later after scripts are done)
* in wrn16x4 exps its 200 epochs, in wrn28x10 its 220 to see whats happens next.

## more thoughts
* is gap aware even important in 2 partitions setting? staleness is just 1.
* however this can allow more staleness (if its even needed).
* note we don't really need the weight stashing to preform gap aware here, its just `lr*grad`.
* however, weight stashing can give us "less noise" (i.e gap) between fwd and backward, but this does not matter much in 2 partitions.
* gap aware + weight stashing (without msnag) doesn't make much sense, its just for comparing (but so far its the best result for 4 partitions!)

## TODOs:
*  I didn't use "weight stashing just for statistics" because we can know the gap norm from straight from the grad norm. (just multiply it by the current lr).
* plot gap from grad norm. (note its tricky: got the warmup and 3 drops, moved a little by warmup).
* sequential runs.
