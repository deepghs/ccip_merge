import json
import os
from functools import lru_cache
from typing import List, Dict, Iterator

import numpy as np
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_file_to_file
from huggingface_hub import HfFileSystem

hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))

SRC_REPO = 'deepghs/character_index'


@lru_cache()
def _get_source_list() -> List[dict]:
    return json.loads(hf_fs.read_text(f'datasets/{SRC_REPO}/characters.json'))


@lru_cache()
def _get_source_dict() -> Dict[str, dict]:
    return {item['tag']: item for item in _get_source_list()}


def list_character_tags() -> Iterator[str]:
    for item in _get_source_list():
        yield item['tag']


def get_detailed_character_info(tag: str) -> dict:
    return _get_source_dict()[tag]


def get_np_feats(tag):
    item = get_detailed_character_info(tag)
    with TemporaryDirectory() as td:
        np_file = os.path.join(td, 'feat.npy')
        download_file_to_file(
            repo_id=SRC_REPO,
            repo_type='dataset',
            file_in_repo=f'{item["hprefix"]}/{item["short_tag"]}/feat.npy',
            local_file=np_file,
        )
        return np.load(np_file)
