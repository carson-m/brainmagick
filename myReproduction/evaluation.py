# do segment level evaluation, get top-10 and top-1 accuracy
import torch

class SegmentLevel_Eval():
    def __init__(self, all_embeddings):
        self.all_embeddings = all_embeddings # (B, C, T)