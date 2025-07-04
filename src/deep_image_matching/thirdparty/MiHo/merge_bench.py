import os
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
import src.AdaLAM.adalam_custom as adalam
import src.DeDoDe2.dedode2_custom as dedode2
import src.DeMatch.dematch_custom as dematch
import src.CLNet.clnet_custom as clnet
import src.FCGNN.fcgnn_custom as fcgnn
import src.MS2DGNet.ms2dgnet_custom as ms2dgnet
import src.NCMNet.ncmnet_custom as ncmnet
import src.bench_utils as bench
import src.ConvMatch.convmatch_custom as convmatch
import src.ConsensusClustering.consensusclustering_custom as consensusclustering

# from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
# from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
# from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
# from src.DIM_modules.loftr_module import loftr_module


if __name__ == '__main__':    
    # available RANSAC: pydegensac, magsac, poselib        

    pipe_head = lambda: None
    pipe_head.placeholder = 'head'

    pipe_ransac = lambda: None
    pipe_ransac.placeholder = 'ransac'

    pipes = [
        [
            pipe_head,
            pipe_ransac
        ],

        [
            pipe_head,
            ncc.ncc_module(also_prev=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True),            
            pipe_ransac
        ],
    
        [
            pipe_head,
            gms.gms_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            oanet.oanet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            adalam.adalam_module(),
            pipe_ransac
        ],                  
        
        [
            pipe_head,
            acne.acne_module(),
            pipe_ransac
        ],        

        [
            pipe_head,
            consensusclustering.consensusclustering_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            dematch.dematch_module(),
            pipe_ransac
        ], 

        [
            pipe_head,
            convmatch.convmatch_module(),
            pipe_ransac
        ], 

        [
            pipe_head,
            fcgnn.fcgnn_module(),
            pipe_ransac
        ],  

        [
            pipe_head,
            clnet.clnet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            ms2dgnet.ms2dgnet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            ncmnet.ncmnet_module(),
            pipe_ransac
        ],            
    ]

    pipe_heads = [
        pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
        pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True),     
        pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint'),
        pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked'),
        pipe_base.lightglue_module(num_features=8000, upright=True, what='disk'),  
        pipe_base.loftr_module(num_features=8000, upright=True),        
        dedode2.dedode2_module(num_features=8000, upright=True),                
        # superpoint_lightglue_module(nmax_keypoints=8000),
        # aliked_lightglue_module(nmax_keypoints=8000),
        # disk_lightglue_module(nmax_keypoints=8000),
        # loftr_module(nmax_keypoints=8000),  
        ]
    
    pipe_ransacs = [
        pipe_base.magsac_module(px_th=1.00),
        pipe_base.magsac_module(px_th=0.75),
        ]
    
    for pipe_module in pipe_heads: pipe_module.placeholder = 'head'
    for pipe_module in pipe_ransacs: pipe_module.placeholder = 'ransac'    

    pipe_save_to = [pipe_head.get_id() for pipe_head in pipe_heads]

###

    split_path = 'split'  # contain as subfolders the bench_data/<bench_res> to be merger
    bench_path = 'merged' # the bench_data/<bench_res> folder of the merged data
    bench_res = 'res'
    save_to = bench_res
    force_list = False    # force list recomputation for data already processed
    force_merge = False   # force generation for data already processed
    
    essential_th = [0.5]
    split = os.listdir(split_path)    
    
    os.makedirs(bench_path, exist_ok=True)
    split_list_file = os.path.join(bench_path, 'split_list.pbz2')
    if os.path.isfile(split_list_file) and (not force_list):
        split_file, split_data = bench.decompress_pickle(split_list_file)
    else:    
        split_file = []
        split_data = []
        
    for d in split:
        dd = os.path.join(split_path,d)
        eval_file = os.listdir(dd)
        for f in eval_file:
            if f[-4:] == 'pbz2':
                ff = os.path.join(dd, f)
                
                if not (ff in split_file):  
                    print(f'processing: {ff}')
                    kk = bench.decompress_pickle(ff).keys()             
                    for k in kk:
                        split_file.append(ff)
                        split_data.append(k + '$')
                    bench.compressed_pickle(split_list_file, (split_file, split_data))
                else:
                    print(f'skipping: {ff}')

    print("*** file list done ***")
                        
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
    for b in benchmark_data.keys():
        # print("*** " + benchmark_data[b]['Name'] + " ***")
        
        # b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
        
        if benchmark_data[b]['is_not_planar']:
            bench_mode = 'fundamental_matrix'
            to_retain = ['fundamental', 'essential', 'match_count']
        else:
            bench_mode = 'homography'
            to_retain = ['homography', 'match_count']   
            
        for ip in range(len(pipe_heads)):
            pipe_head = pipe_heads[ip]
            
            # print("*** " + pipe_head.get_id() + " ***")
            
            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + pipe_save_to[ip] + '_')
            to_save_file_suffix ='_' + benchmark_data[b]['name']
            
            working_dict = {}
                        
            for jp in range(len(pipe_ransacs)):
                pipe_ransac = pipe_ransacs[jp]

                # print("*** " + pipe_ransac.get_id() + " ***")
                
                for i, pipe in enumerate(pipes):                                        
                    # print(f"*** Pipeline {i+1}/{len(pipes)} ***")        

                    for k, pipe_module in enumerate(pipe):
                        if hasattr(pipe_module, 'placeholder'):
                            if pipe_module.placeholder == 'head': pipe[k] = pipe_head
                            if pipe_module.placeholder == 'ransac': pipe[k] = pipe_ransac

                    for pipe_module in pipe:
                        if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', bench_mode)
                        if hasattr(pipe_module, 'outdoor'): setattr(pipe_module, 'outdoor', benchmark_data[b]['is_outdoor'])

                    for rr in to_retain:
                        if not (rr in working_dict.keys()):
                            working_dict[rr] = {}
                        
                        split_file_ = [s for s in split_file if rr in s]                        
                        split_data_ = [s2 for s1, s2 in zip(split_file, split_data) if rr in s1]                        

                        for essn, ess_th in enumerate(essential_th):
    
                            if rr != 'essential' and (essn > 0):                        
                                continue
    
                            pipe_name_base = os.path.join(bench_res, benchmark_data[b]['name'])                                
                            for pipe_module in pipe:
                                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())

                                if rr == 'essential':
                                    pipe_name_base_ok = pipe_name_base + '_essential_th_list_' + str(ess_th)
                                else:
                                    pipe_name_base_ok = pipe_name_base

                                ll = [[b, a] for a, b in zip(split_data_, split_file_) if pipe_name_base_ok + '$'  in a]
                                ll.sort()
                                
                                if len(ll) == 0:
                                    print(f'missing: {pipe_name_base_ok}')
                                    continue
                                
                                to_open = ll[-1][0]
                                to_dict = ll[-1][1]
                                
                                if not(to_open in working_dict[rr].keys()):
                                    working_dict[rr][to_open] = []
                                    
                                isin = False
                                for cc in working_dict[rr][to_open]:
                                    if cc == to_dict:
                                        isin = True
                                        break
    
                                if not isin:
                                    working_dict[rr][to_open].append(to_dict)
                            
            for rr in to_retain:
                save_to_ = to_save_file + rr + to_save_file_suffix + '.pbz2' 
                eval_data = {}
                
                if (not os.path.isfile(save_to_)) or force_merge:   
                    print(f'generating: {save_to_}')
                    
                    for kk in working_dict[rr].keys():
                        old_eval = bench.decompress_pickle(kk)
                        for vv in  working_dict[rr][kk]:                            
                            base_name = vv[:-1]
                            base_name = base_name[base_name.rfind(pipe_head.get_id()):]
                            
                            eval_data[base_name] = old_eval[vv[:-1]]
    
                    os.makedirs(os.path.split(save_to_)[0], exist_ok=True)
                    bench.compressed_pickle(save_to_, eval_data)    
                else:
                    print(f'skipping: {save_to_}')

            if benchmark_data[b]['is_not_planar']:
                bench.csv_summary_non_planar(essential_th_list=[0.5], essential_load_from=to_save_file + 'essential' + to_save_file_suffix + '.pbz2', fundamental_load_from=to_save_file + 'fundamental' + to_save_file_suffix + '.pbz2', save_to=to_save_file + 'fundamental_and_essential' + to_save_file_suffix + '.csv', match_count_load_from=to_save_file + 'match_count' + to_save_file_suffix + '.pbz2', also_metric=benchmark_data[b]['also_metric'], to_remove_prefix=pipe_head.get_id())
            else:
                bench.csv_summary_planar(load_from=to_save_file + 'homography' + to_save_file_suffix + '.pbz2', save_to=to_save_file + 'homography' + to_save_file_suffix + '.csv', match_count_load_from=to_save_file + 'match_count' + to_save_file_suffix + '.pbz2', to_remove_prefix=pipe_head.get_id())
