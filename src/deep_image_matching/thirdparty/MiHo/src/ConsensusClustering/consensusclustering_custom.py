import numpy as np

try:
    import pymulticonsensus
    pymulticonsensus_off = True
except:
    pymulticonsensus_off = False
    import warnings
    warnings.warn("cannot load pymulticonsensus - Consensus Clustering (CC) module will return no matches")

from PIL import Image
from .utils import model_type


class consensusclustering_module:
    def __init__(self, **args):
        self.inlier_th = 7 # 1.5 in the authors' git example
        self.maximum_tanimoto_similarity = 0.75
        self.cluster_assign = 'hard'
        self.confidence = 0.999
        self.starting_hypothesis_number = 10
        self.added_hypothesis_number = 10
        self.max_iters = 75
        self.minimum_point_number = 12
        self.sampler_id = 1
        self.hard_assign_neighborhood_size = 20.0
        self.hard_assign_spatial_coherence_weight = 0.0
                
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('consensusclustering_th_' + str(self.inlier_th) + '_max_tanimoto_similarity_' + str(self.maximum_tanimoto_similarity) + '_cluster_assignment_' + self.cluster_assign).lower()


    if pymulticonsensus_off:
        def run(self, **args):                
            mask = []
            pt1 = args['pt1']
            pt2 = args['pt2']     
            Hs = args['Hs']
                       
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
    else:
        def run(self, **args):        
            k1 = np.ascontiguousarray(args['pt1'].detach().cpu())
            k2 = np.ascontiguousarray(args['pt2'].detach().cpu())
        
            if k1.shape[0] <= 4:
                mask = []
                pt1 = args['pt1']
                pt2 = args['pt2']     
                Hs = args['Hs']
           
                return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
        
            correspondences = np.ascontiguousarray(np.concatenate([k1, k2], axis=1))      
        
            sz1 = Image.open(args['im1']).size
            sz2 = Image.open(args['im2']).size        
        
            Hs = pymulticonsensus.findHomographies(correspondences, sz1[1], sz1[0], sz2[1], sz2[0], threshold = self.inlier_th, confidence = 0.999, maximum_tanimoto_similarity=self.maximum_tanimoto_similarity, starting_hypothesis_number=self.starting_hypothesis_number, added_hypothesis_number=self.added_hypothesis_number, max_iters=self.max_iters, minimum_point_number=self.minimum_point_number, sampler_id=self.sampler_id)          
            Hs = np.asarray(Hs)
        
            n_models = int(Hs.shape[0] / 3)
        
            if n_models > 0:        
                if self.cluster_assign == 'soft':
                    soft_assignment = pymulticonsensus.getSoftLabeling(Hs.reshape(n_models, 9), correspondences, model_type("homography"), self.inlier_th)                
                    mask = np.max(soft_assignment, axis=1) > 0.01
                else:
                    labeling = pymulticonsensus.getLabeling(Hs.reshape(n_models, 9), correspondences, model_type("homography"), self.inlier_th, self.hard_assign_neighborhood_size, self.hard_assign_spatial_coherence_weight, self.minimum_point_number)
                    mask = labeling > -1
                    
                pt1 = args['pt1'][mask]
                pt2 = args['pt2'][mask]     
                Hs = args['Hs'][mask]
            else:            
                mask = []
                pt1 = args['pt1']
                pt2 = args['pt2']     
                Hs = args['Hs']
            
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
