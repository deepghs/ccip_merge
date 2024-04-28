import time

import numpy as np
import pandas as pd
from ditk import logging
from imgutils.metrics import ccip_batch_differences, ccip_default_threshold
from tqdm import tqdm

from .index import get_np_feats
from .picked import PICKED_TAGS


def measure_tag_via_func(tag, func):
    logging.info(f'Reading embeddings for tag {tag!r} ...')
    embs = get_np_feats(tag)
    logging.info(f'Embedding shape: {embs.shape!r} ...')

    logging.info('Merging embeddings ...')
    start_time = time.time()
    result_emb = func(embs)
    duration = time.time() - start_time

    logging.info(f'Result embedding shape of {tag!r}: {result_emb.shape!r}.')
    distances = ccip_batch_differences([result_emb, *embs])[0, 1:]

    retval = {
        'mean_diff': distances.mean(),
        'same_ratio': (distances < ccip_default_threshold()).mean(),
        'time_cost': duration,
    }
    logging.info(f'Tag {tag!r}, mean diff: {retval["mean_diff"]:.4f}, '
                 f'same ratio: {retval["same_ratio"]:.4f}, time cost: {retval["time_cost"]:.4f}s.')
    return retval


def ccip_merge_func(embs):
    lengths = np.linalg.norm(embs, axis=-1)
    embs = embs / lengths.reshape(-1, 1)
    ret_embedding = embs.mean(axis=0)
    return ret_embedding / np.linalg.norm(ret_embedding) * lengths.mean()


def get_metrics_of_tags(n: int = 100) -> pd.DataFrame:
    rows = []
    for tag in tqdm(PICKED_TAGS[:n]):
        logging.info(f'Merging for tag {tag!r} ...')
        metrics = measure_tag_via_func(tag, ccip_merge_func)
        rows.append({'tag': tag, **metrics})

    return pd.DataFrame(rows)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    df = get_metrics_of_tags(n=100)
    logging.info(str(df))
    logging.info(f'Mean diff: {df["mean_diff"].mean():.4f}, '
                 f'same ratio: {df["same_ratio"].mean():.4f}, '
                 f'time cost: {df["time_cost"].mean():.4f}s.')

    file = 'test_result.csv'
    logging.info(f'Saving result to {file!r} ...')
    df.to_csv(file, index=False)
