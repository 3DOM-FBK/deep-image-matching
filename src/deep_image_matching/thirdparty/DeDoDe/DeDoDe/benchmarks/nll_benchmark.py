import torch
import torch.nn as nn
from DeDoDe.utils import *
import DeDoDe

class MegadepthNLLBenchmark(nn.Module):
    
    def __init__(self, dataset, num_samples = 1000, batch_size = 8, device = "cuda") -> None:
        super().__init__()
        sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(dataset)), replacement=False, num_samples=num_samples
            )
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=batch_size, sampler=sampler
            )
        self.dataloader = dataloader
        self.tracked_metrics = {}
        self.batch_size = batch_size
        self.N = len(dataloader)    
    
    def compute_batch_metrics(self, detector, descriptor, batch, device = "cuda"):
        kpts = detector.detect(batch)["keypoints"]
        descriptions_A, descriptions_B = descriptor.describe_keypoints(batch, kpts)["descriptions"].chunk(2)
        kpts_A, kpts_B = kpts.chunk(2)
        mask_A_to_B, kpts_A_to_B = warp_kpts(kpts_A, 
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],)
        mask_B_to_A, kpts_B_to_A = warp_kpts(kpts_B, 
                    batch["im_B_depth"],
                    batch["im_A_depth"],
                    batch["T_1to2"].inverse(),
                    batch["K2"],
                    batch["K1"],)
        with torch.no_grad():
            D_B = torch.cdist(kpts_A_to_B, kpts_B)
            D_A = torch.cdist(kpts_A, kpts_B_to_A)
            inds = torch.nonzero((D_B == D_B.min(dim=-1, keepdim = True).values) 
                                 * (D_A == D_A.min(dim=-2, keepdim = True).values)
                                 * (D_B < 0.01)
                                 * (D_A < 0.01))
        logP_A_B = dual_log_softmax_matcher(descriptions_A, descriptions_B, 
                                            normalize = True,
                                            inv_temperature = 20)
        neg_log_likelihood = -logP_A_B[inds[:,0], inds[:,1], inds[:,2]].mean()
        self.tracked_metrics["neg_log_likelihood"] = self.tracked_metrics.get("neg_log_likelihood", 0) + 1/self.N * neg_log_likelihood

    def benchmark(self, detector, descriptor):
        self.tracked_metrics = {}
        from tqdm import tqdm
        print("Evaluating percent inliers...")
        for idx, batch in tqdm(enumerate(self.dataloader), mininterval = 10.):
            batch = to_cuda(batch)
            self.compute_batch_metrics(detector, descriptor, batch)
        [print(name, metric.item() * self.N / (idx+1)) for name, metric in self.tracked_metrics.items()]