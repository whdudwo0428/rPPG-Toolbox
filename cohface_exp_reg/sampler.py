# -*- coding: utf-8 -*-
import random
from typing import List, Dict

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[List[int]]):
    """Group by sequence length and yield batches based on bucket→batchsize map.
    expects dataset.idxs = (sid, start, T)
    """

    def __init__(self, dataset, bucket_bs: Dict[int, int], shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        # collect indices per T
        perT = {}
        for i, (_sid, _st, T) in enumerate(dataset.idxs):
            perT.setdefault(T, []).append(i)
        self.perT = perT
        self.bucket_bs = bucket_bs

    def __iter__(self):
        all_batches = []
        for T, idxs in self.perT.items():
            bs = self.bucket_bs.get(T, 4)
            idxs = idxs[:]
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), bs):
                all_batches.append(idxs[i:i + bs])
        if self.shuffle:
            random.shuffle(all_batches)
        for b in all_batches:
            yield b

    def __len__(self):
        n = 0
        for T, idxs in self.perT.items():
            bs = self.bucket_bs.get(T, 4)
            n += (len(idxs) + bs - 1) // bs
        return n