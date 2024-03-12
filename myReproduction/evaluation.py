# do segment level evaluation, get top-10 and top-1 accuracy
import torch

class SegmentLevel_Eval():
    def __init__(self, all_embeddings):
        self.all_embeddings = all_embeddings # (B, C, T)
    
    def get_accuracy(self, estimates, true_labels, top_k):
        # (B, C, T) * (B', C, T) -> (B, B')
        self.all_embeddings = self.all_embeddings.to(estimates)
        scores = torch.einsum("bct, oct->bo", estimates, self.all_embeddings)
        __, topk_indices = torch.topk(scores, top_k, dim = -1)
        true_labels = true_labels.to(topk_indices)
        topk_indices -= true_labels.view(-1, 1)
        correct = torch.sum(topk_indices == 0)
        return correct / len(true_labels)
    
def test():
    all_embeddings = torch.rand(100, 320, 100)
    eval = SegmentLevel_Eval(all_embeddings)
    estimates = torch.rand(5, 320, 100)
    true_labels = torch.randint(0, 10, (5,))
    top_10_acc = eval.get_accuracy(estimates, true_labels, 10)
    top_1_acc = eval.get_accuracy(estimates, true_labels, 1)
    print(top_10_acc, top_1_acc)
    
if __name__ == "__main__":
    test()