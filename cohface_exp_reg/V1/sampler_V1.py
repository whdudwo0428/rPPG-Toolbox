# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    """길이별로 다른 배치 크기 적용, 패딩 낭비 최소화"""
    def __init__(self, lengths, bucket_bs_map):
        super().__init__()
        self.lengths = np.asarray(lengths)
        self.buckets = defaultdict(list)
        for idx, L in enumerate(self.lengths):
            self.buckets[L].append(idx)
        self.bucket_bs_map = bucket_bs_map

        self._batches = []
        for L, idxs in self.buckets.items():
            bs = self.bucket_bs_map.get(L, 8)
            for i in range(0, len(idxs), bs):
                self._batches.append(idxs[i:i+bs])

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        # 셔플은 상위 DataLoader의 shuffle=False로 두고 여기서 섞어도 됨
        rng = np.random.default_rng()
        order = np.arange(len(self._batches))
        rng.shuffle(order)
        for i in order:
            yield self._batches[i]
