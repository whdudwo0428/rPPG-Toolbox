# cohface_exp_reg/sampler.py
import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths, bucket_bs, shuffle=True):
        super().__init__()
        self.lengths = list(lengths)
        self.bucket_bs = dict(bucket_bs)
        self.shuffle = shuffle
        buckets = defaultdict(list)
        for idx, T in enumerate(self.lengths):
            buckets[int(T)].append(idx)
        self.buckets = dict(buckets)
        missing = [T for T in self.buckets.keys() if T not in self.bucket_bs]
        if missing:
            keys = sorted(self.bucket_bs.keys())
            for T in missing:
                self.bucket_bs[T] = min(self.bucket_bs.values())

    def __iter__(self):
        all_batches = []
        for T, idxs in self.buckets.items():
            if self.shuffle:
                random.shuffle(idxs)
            bs = max(1, int(self.bucket_bs.get(T, 1)))
            for i in range(0, len(idxs), bs):
                batch = idxs[i:i+bs]
                if len(batch) > 0:
                    all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        for b in all_batches:
            yield b

    def __len__(self):
        n = 0
        for T, idxs in self.buckets.items():
            bs = max(1, int(self.bucket_bs.get(T, 1)))
            n += math.ceil(len(idxs)/bs)
        return n

def parse_bucket_bs(s: str) -> dict:
    out = {}
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split(":")
        out[int(k)] = int(v)
    return out
