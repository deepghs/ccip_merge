# ccip_merge

Merge algorithm of ccip embeddings

## Install

```shell
git clone https://github.com/narugo1992/ccip_merge.git
cd ccip_merge
pip install -r requirements.txt
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
