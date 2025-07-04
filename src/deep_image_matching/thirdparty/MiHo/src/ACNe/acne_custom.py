import os

import uuid
import numpy as np
import warnings
from PIL import Image
import zipfile
import gdown

from .demo_util import get_T_from_K, norm_points_with_T
from .utils import norm_points
from .network import MyNetwork
from .config import get_config

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def prepare_xs(xs, K1, K2, use_fundamental=0):
    """
    Prepare xs for model
    Inputs:
        xs: Nx4, Input correspondences in original image coordinates
        K: 3x3, Calibration matrix
        use_fundamental:
            0 means E case, xs is calibrated; 
            1 means F case, xs is normalized with mean, std of coordinates;
            2 means F case, xs is normalized with image size.
    Returns:
        xs: Nx4, Input correspondences in calibrated (E) or normed (F) coordinates
        T: 3x3, Transformation matrix used to normalize input in the case of F. None in the case of E.
    """
    x1, x2 = xs[:,:2], xs[:,2:4]
    if use_fundamental>0:
        # Normalize Points
        if use_fundamental == 1:
            # normal normalization
            x1, T1 = norm_points(x1)
            x2, T2 = norm_points(x2)
        elif use_fundamental == 2:
            # we used img_size normization
            T1 = get_T_from_K(K1)
            T2 = get_T_from_K(K2)
            x1 = norm_points_with_T(x1, T1)
            x2 = norm_points_with_T(x2, T2)
        else:
            raise NotImplementedError
        xs = np.concatenate([x1,x2],axis=-1).reshape(-1,4)
    else:
        # Calibrate Points with intrinsics
        x1 = (
            x1 - np.array([[K1[0,2], K1[1,2]]])
            ) / np.array([[K1[0,0], K1[1,1]]])
        x2 = (
            x2 - np.array([[K2[0,2], K2[1,2]]])
            ) / np.array([[K2[0,0], K2[1,1]]])
        xs = np.concatenate([x1,x2],axis=-1).reshape(-1,4)
        T1, T2 = None, None

    return xs, T1, T2

class NetworkTest(MyNetwork):

    def __init__(self, config, model_path):
        super(NetworkTest, self).__init__(config)
        
        # restore from model_path
        self.saver_best.restore(self.sess, model_path)
        
    def compute_E(self, xs):
        """
        Compute E/F given a set of putative correspondences. The unit weight vector 
        for each correspondenece is also given.
        Input:
            xs: BxNx4
        Output:
            out_e: Bx3x3
            w_com: BxN
            socre_local: BxN
        """
        _xs = np.array(xs).reshape(1, 1, -1, 4)
        feed_dict = {
            self.x_in: _xs,
            self.is_training: False, 
        }
        fetch = {
            "w_com": self.last_weights,
            "score_local":self.last_logit,
            "out_e": self.out_e_hat
        }
        res = self.sess.run(fetch, feed_dict=feed_dict)
        batch_size = _xs.shape[0]
        score_local = res["score_local"] # 
        w_com = res["w_com"] # Combined weights
        out_e = res["out_e"].reshape(batch_size, 3, 3)
        return out_e, w_com, score_local
        

class acne_module:
    current_net = None
    current_obj_id = None
    
    def __init__(self, **args):
        acne_dir = os.path.split(__file__)[0]
        model_dir = os.path.join(acne_dir, 'logs')

        file_to_download = os.path.join(acne_dir, 'acne_weights.zip')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/1yluw3u3F8qH3oTB3dxVw1re4HI6a0TuQ/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)        

        file_to_unzip = file_to_download
        if not os.path.isdir(model_dir):    
            with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
                zip_ref.extractall(path=acne_dir)

        self.outdoor = True
        self.prev_outdoor = True

        for k, v in args.items():
           setattr(self, k, v)
           
        if self.outdoor:
            # Model of ACNe_F trained with outdoor dataset.              
            model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
        else:
            # Model of ACNe_F trained with indoor dataset.                      
             model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"        

        self.acne_id = uuid.uuid4()
             
        acne_dir = os.path.split(__file__)[0]
        self.model_path = os.path.join(acne_dir, model_path)
                           
    def get_id(self):
        return ('acne_outdoor_' + str(self.outdoor)).lower()

        
    def run(self, **args):
        
        force_reload = False
        if (self.outdoor != self.prev_outdoor):
            force_reload = True
            self.prev_outdoor = self.outdoor
            
            if self.outdoor:
                # Model of ACNe_F trained with outdoor dataset.              
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
            else:
                # Model of ACNe_F trained with indoor dataset.                      
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"
            acne_dir = os.path.split(__file__)[0]
            self.model_path = os.path.join(acne_dir, model_path)


        if (acne_module.current_obj_id != self.acne_id) or force_reload:
            if not (acne_module.current_obj_id is None):
                acne_module.current_net.sess.close()
                tf.reset_default_graph()

            warnings.filterwarnings('ignore')
            config, unparsed = get_config()
        
            paras = {
                "CNe_E":{
                    "bn_opt":"bn"},
                "ACNe_E":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax"},
                "CNe_F":{
                    "bn_opt":"bn", "use_fundamental":2},
                "ACNe_F":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax", "use_fundamental":2},
            }
        
            para = paras["ACNe_F"]
    
            for ki, vi in para.items():
               setattr(config, ki, vi)
               
            self.use_fundamental = config.use_fundamental # E:0, F:2.
        
            # Instantiate wrapper class
            self.net = NetworkTest(config, self.model_path)
            acne_module.current_net = self.net
            acne_module.current_obj_id = self.acne_id
            
        pt1 = np.ascontiguousarray(args['pt1'].detach().cpu())
        pt2 = np.ascontiguousarray(args['pt2'].detach().cpu())

        sz1 = Image.open(args['im1']).size
        sz2 = Image.open(args['im2']).size        
                
        l = pt1.shape[0]
        
        if l > 0:    
            corrs = np.hstack((pt1, pt2)).astype(np.float32)
        
            K1=np.array(
                [[1, 0, sz1[0]/2.0],
                 [0, 1, sz1[1]/2.0],
                 [0, 0 ,1]])
        
            K2=np.array(
                [[1, 0, sz2[0]/2.0],
                 [0, 1, sz2[1]/2.0],
                 [0, 0, 1]])
        
            # Prepare input. 
            xs, T1, T2 = prepare_xs(corrs, K1, K2, self.use_fundamental)
            xs = np.array(xs).reshape(1, 1, -1, 4) # reconstruct a batch. Bx1xNx4
        
            # Compute Essential/Fundamental matrix
            E, w_com, score_local = self.net.compute_E(xs)
            E = E[0]
            score_local = score_local[0]
            w_com = w_com[0]
        
            mask = w_com > 1e-5
        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]            
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
