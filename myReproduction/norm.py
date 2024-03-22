# clamping of the EEG signals
import torch

class RobustScaler():
    """
    Similar to RobustScaler for sklearn but can run on GPU
    """
    def __init__(self, lowq=0.25, highq=0.75, subsample=1., device="cpu"):
        self.lowq = lowq
        self.highq = highq
        self.subsample = subsample
        self.device = device

    def transform(self, X):
        dimension, samples = X.shape
        scale_ = torch.empty(dimension)
        center_ = torch.empty(dimension)
        X = X.to(self.device)
        for d in range(dimension):
            # to dimension one at a time to fit on the GPU memory
            col = X[d, :]
            keep = torch.rand_like(col) < self.subsample
            col = col[keep]
            # torch 1.7.0 has a quantile function but it is not really faster than sorting.
            col, _ = col.sort()
            quantiles = [self.lowq, 0.5, self.highq]
            low, med, high = [col[int(q * len(col))].item() for q in quantiles]
            # print(low, med, high)
            scale_[d] = high - low
            center_[d] = med
            if scale_[d] == 0:
                # this will happen as we are padding some recordings
                # so that all recordings have the same number of channels.
                scale_[d] = 1
        assert (scale_ != 0).any()
        scale_[scale_ == 0] = 1
        return (X - center_.to(X)[:,None]) / scale_.to(X)[:,None]

class StandardNorm:
    def __init__(self, device = "cpu"):
        self.device = device
    
    def transform(self, X):
        X = X.to(self.device)
        mean = X.mean()
        # !!!!! to be fixed
    
# def test():
#     X = torch.tensor([[1,2,3,4,5,6,7,8,9],[1,1,1,1,1,1,1,1,1]]).float()
#     print('X=', X)
#     print(X.shape)
#     sca = RobustScaler()
#     print(sca.transform(X))

def test():
    X = torch.tensor([[1,2],[1,2]]).float()
    std = StandardNorm()
    print(std.transform(X))
    
if __name__ == "__main__":
    test()