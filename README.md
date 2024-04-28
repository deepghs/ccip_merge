# ccip_merge

Merge algorithm of ccip embeddings

## Install

```shell
git clone https://github.com/narugo1992/ccip_merge.git
cd ccip_merge
pip install -r requirements.txt
```

## Best Solution

**New update in 2024.4.28: Solved, the best solution is just aligning the norm of embeddings, and get their center
point.** Like this

```python
import numpy as np


def ccip_merge_func(embs):
    lengths = np.linalg.norm(embs, axis=-1)
    embs = embs / lengths.reshape(-1, 1)
    ret_embedding = embs.mean(axis=0)
    return ret_embedding / np.linalg.norm(ret_embedding) * lengths.mean()

```

## Run Experiment

**Target: FIND OUT AN EMBEDDING CLOSEST TO THE GIVEN EMBEDDINGS**

Go and change your code in `ccip_merge/measure.py`

```python
def ccip_merge_func(embs):
    # TODO: just replace your function here!!!
    pass
```

You need to output a merge embedding (shape: `(768, )`) based on the input embeddings (shape: `(n, 768)`).
And then the merged embedding will be used to calculate the distance between the input ones
with `imgutils.metrics.ccip_batch_differences` function.

The current `ccip_merge_func` is a default one based on `scipy.optimize.minimize`. **We think you must be able to write
a
better one.**

After you complete these, just run

```shell
python -m ccip_merge.measure
```

100 randomly selected fixed character tags will be used as the test cases.
And the result will be saved at `test_result.csv`.
