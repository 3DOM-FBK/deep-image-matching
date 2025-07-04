import os
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
import src.AdaLAM.adalam_custom as adalam
import src.DeMatch.dematch_custom as dematch
import src.ConvMatch.convmatch_custom as convmatch
import src.DeDoDe2.dedode2_custom as dedode2
import src.FCGNN.fcgnn_custom as fcgnn
import src.CLNet.clnet_custom as clnet
import src.NCMNet.ncmnet_custom as ncmnet
import src.MS2DGNet.ms2dgnet_custom as ms2dgnet
import src.ConsensusClustering.consensusclustering_custom as consensusclustering
import src.bench_utils as bench
import numpy as np
import os
import shutil

# from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
# from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
# from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
# from src.DIM_modules.loftr_module import loftr_module


def csv_write(lines, save_to='nameless.csv'):

    with open(save_to, 'w') as f:
        for l in lines:
            f.write(l)   


def compile_latex(latex_file):
    # require pdflatex to be installed

    os.makedirs('tmp', exist_ok=True)
    shutil.copy(latex_file, 'tmp/aux.tex') 
    os.system('cd tmp; pdflatex aux.tex')
    os.system('cd tmp; pdflatex aux.tex')
    os.system('export LD_LIBRARY_PATH= && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -dCompressFonts=true -dSubsetFonts=true -dColorConversionStrategy=/LeaveColorUnchanged -dPrinted=false -sOutputFile=tmp/aux_.pdf tmp/aux.pdf');
    os.system('pdfcrop tmp/aux_.pdf tmp/aux__.pdf')
    shutil.copy('tmp/aux__.pdf', latex_file[:-4] + '.pdf');
    os.system('rm -R tmp');


def csv_merger(csv_list, include_match_count=False):

    if not include_match_count:
        avg_idx = [[ 3,  6, 'F_AUC@avg_a'], # MegaDepth
                   [ 6,  9, 'E_AUC@avg_a'],
                   [11, 14, 'F_AUC@avg_a'], # ScanNet
                   [14, 17, 'E_AUC@avg_a'],
                   [19, 22, 'H_AUC@avg_m'], # Planar
                   [24, 27, 'F_AUC@avg_a'], # PhotoTourism
                   [27, 30, 'F_AUC@avg_m'],
                   [30, 33, 'E_AUC@avg_a'],
                   [33, 36, 'E_AUC@avg_m'],
                   ]
    else:           
        avg_idx = [[ 4,  7, 'F_AUC@avg_a'], # MegaDepth
                   [ 7, 10, 'E_AUC@avg_a'], 
                   [13, 16, 'F_AUC@avg_a'], # ScanNet
                   [16, 19, 'E_AUC@avg_a'],
                   [21, 25, 'H_AUC@avg_m'], # Planar
                   [28, 31, 'F_AUC@avg_a'], # PhotoTourism
                   [31, 34, 'F_AUC@avg_m'],
                   [34, 37, 'E_AUC@avg_a'],
                   [37, 40, 'E_AUC@avg_m'],
                   ]               
        
    csv_data = []
    for csv_file in csv_list:
        aux = [csv_line.split(';') for csv_line in  open(csv_file, 'r').read().splitlines()]
        to_fuse = max([idx for idx, el in enumerate([s.startswith('pipe_module') for s in aux[0]]) if el == True]) + 1

        tmp = {}
        for row in aux:
            what = ';'.join(row[:to_fuse]).replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
            tmp[what] = row[to_fuse:]

        csv_data.append(tmp)
    
    pipe_set = {}
    for k in csv_data:
        for w in csv_data[0].keys():
            pipe_set[w] = '0'
        
    merged_csv = []
    for k in pipe_set.keys():        
        row = [k]
        for curr_csv in csv_data:
            if k in curr_csv:            
                 to_add = [el for el in curr_csv[k]]
            else:
                to_add = ['nan' for el in curr_csv[list(curr_csv.keys())[0]]]                
            row.extend(to_add)
        merged_csv.append(row)
        
    trimmed_avg_idx = []
    for avg_i in avg_idx:
        if avg_i[1] <= len(row):
            trimmed_avg_idx.append(avg_i)
    
        
    avg_csv = []
    for row in merged_csv:        
        if 'pipe_module' in row[0]:
            avg_list = [rrange[2] for rrange in trimmed_avg_idx]
        else:
            avg_list = [np.mean([float(i) for i in row[rrange[0]:rrange[1]]]) for rrange in trimmed_avg_idx]
        avg_csv.append(avg_list)

    fused_csv = []
    for row_base, row_avg in zip(merged_csv, avg_csv):
        row_new =  []
        for k in range(len(trimmed_avg_idx) - 1, - 1, - 1):
            if k == 0:
                l = 0
            else:
                l = trimmed_avg_idx[k - 1][1]
                
            if k == len(trimmed_avg_idx) - 1:
                r = len(row_base)
            else:
                r = trimmed_avg_idx[k][1]
                               
            row_new =  row_base[l:r] + [str(row_avg.pop())] + row_new 
        fused_csv.append(row_new)
        
    only_num_csv = [row[1:] for row in fused_csv[1:]]
    m = np.asarray(only_num_csv, dtype=float)
    sidx = np.argsort(-m, axis=0)
    sidx_ = np.argsort(sidx, axis=0)
    fused_csv_order = np.full((m.shape[0] + 1, m.shape[1] + 1), np.nan)
    fused_csv_order[1:,1:] = sidx_

    return fused_csv, fused_csv_order


def to_latex_simple(csv_table, table_name=''):

    l1 = 1
    lo = 1
    for row in csv_table:
        l1 = max(l1, len(row[0]))
        for i in row[1:]:
            lo = max(lo, len(i))

    header = [
        '\\documentclass[a4paper,10pt]{article}\n',
        '\\usepackage{graphicx}\n',
        '\\usepackage{caption}\n',
        '\\captionsetup{labelformat=empty}\n',
        '\\usepackage{color}\n',
        '\\usepackage{booktabs}\n',
        '\\usepackage{amssymb}\n',
        '\\usepackage[table,usenames,dvipsnames]{xcolor}\n',
        '\\usepackage{amsmath}\n',
        '\\usepackage{ulem}\n',        
        '\\usepackage{calc}\n',        
        '\n',
        '\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\n',        
#       '\\newlength\\MAXA\\setlength\\MAXA{\\widthof{' + ('a' * l1) + 'A.}}\n',
        '\\newlength\\MAXA\\setlength\\MAXA{\\widthof{SuperPoint+LightGlue}}\n',
        '\\newlength\\MAXB\\setlength\\MAXB{\\widthof{' + ('a' * lo) + 'A.}}\n',
        '\n',        
        '\\begin{document}\n', 
        '\\pagestyle{empty}\n',
        '\t\\begin{table}[t!]\n',
        '\t\\centering\n',
#       '\t\t\t\\begin{tabular}{R{\\MAXA}' + ('R{\\MAXB}' * (len(csv_table[0]) - 1)) + '}\n',
        '\t\t\t\\begin{tabular}{R{\\MAXA}' + ('r' * (len(csv_table[0]) - 1)) + '}\n',
    ]
           
    header.append('\t\t\t\t' + ' & '.join(csv_table[0]) + ' \\\\\n')    
    header.append('\t\t\t\t\\midrule\n')

    val_table = [[int(v) for v in row[1:]] for row in csv_table[1:]]
    mmax = np.argmax(np.asarray(val_table), axis=0)
    
    mtable = [csv_table[0]]
    for i, row in enumerate(csv_table[1:]):
        new_row = []
        for j, v in enumerate(row):
            if (j != 0) and (i == mmax[j-1]):
                v = '\\textbf{' + v + '}'
            new_row.append(v)
        mtable.append(new_row)
    
    latex_table = []
    for row in mtable[1:]:
        latex_table.append('\t\t\t\t' + ' & '.join(row) + ' \\\\\n')
           
    footer = [
        '\t\t\t\end{tabular}\n',
        '\t\t\\caption{' + table_name + '}\\label{none}\n',        
        '\t\\end{table}\n',
        '\\end{document}\n',
    ]

    latex_table = header + latex_table + footer
                    
    return latex_table


def to_latex_corr(table_name, ccol, corr_table):

    header = [
        '\\documentclass[a4paper,10pt]{article}\n',
        '\\usepackage{graphicx}\n',
        '\\usepackage{caption}\n',
        '\\captionsetup{labelformat=empty}\n',
        '\\usepackage{color}\n',
        '\\usepackage{booktabs}\n',
        '\\usepackage{amssymb}\n',
        '\\usepackage[table,usenames,dvipsnames]{xcolor}\n',
        '\\usepackage{amsmath}\n',
        '\\usepackage{ulem}\n',        
        '\\usepackage{calc}\n',        
        '\n',
        '\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\n',        
        '\\newlength\\MAX\\setlength\\MAX{\\widthof{Recall.}}\n',
        '\n',        
        '\\begin{document}\n',        
        '\\pagestyle{empty}\n',
        '\t\\begin{table}[t!]\n',
        '\t\\centering\n',
        '\t\t\t\\begin{tabular}{' + ('R{\\MAX}' * (len(ccol)+1)) + '}\n',
    ]
           
    header.append('\t\t\t\t' + ' & ' + ' & '.join(ccol) + ' \\\\\n')    
  # header.append('\t\t\t\t\\midrule\n')

    latex_table = []
    for i in range(len(ccol)):
        row = []
        for j in range(len(ccol)):
            v = "{n:3.2f}".format(n=corr_table[i][j])
            if i>j:
                clr = 'red'
            elif i < j:
                clr = 'blue'
            else:
                clr = 'violet'
                
            v = '\cellcolor{' + clr + '!' + str((corr_table[i][j] + 1) / 2 * 100 * 0.75 + 0.125)  + '}' + v
            row.append(v)

        ccol[i] + ' & ' + ' & '.join(row)
        latex_table.append('\t\t\t\t' + ccol[i] + ' & ' + ' & '.join(row) + ' \\\\\n')
           
    footer = [
        '\t\t\t\end{tabular}\n',
        '\t\t\\caption{Error correlation for the ' + table_name  + ' dataset \\textcolor{blue}{with} and \\textcolor{red}{without} MAGSAC}\\label{none}\n',
        '\t\\end{table}\n',
        '\\end{document}\n',
    ]

    latex_table = header + latex_table + footer
                    
    return latex_table


if __name__ == '__main__':    

    pipes = [
        [     'MAGSAC^', pipe_base.magsac_module(px_th=1.00)],
        [     'MAGSACv', pipe_base.magsac_module(px_th=0.75)],
        [         'NCC', ncc.ncc_module(also_prev=True)],
        [    'MOP+MiHo', miho_duplex.miho_module()],
        [         'MOP', miho_unduplex.miho_module()],
        [         'GMS', gms.gms_module()],
        [       'OANet', oanet.oanet_module()],
        [      'AdaLAM', adalam.adalam_module()],
        [        'ACNe', acne.acne_module()],
        [          'CC', consensusclustering.consensusclustering_module()],
        [     'DeMatch', dematch.dematch_module()],
        [   'ConvMatch', convmatch.convmatch_module()],
        [       'CLNet', clnet.clnet_module()],
        [      'NCMNet', ncmnet.ncmnet_module()],
        [      'FC-GNN', fcgnn.fcgnn_module()],
        ['MS$^2$DG-Net', ms2dgnet.ms2dgnet_module()],


    ]

    pipe_heads = [
          [                                                                   'SIFT+NNR', pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True)],     
          [    'Key.Net+$\\scriptsize\\substack{\\text{AffNet}\\\\\\text{HardNet}}$+NNR', pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99)],
          [                                                       'SuperPoint+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint')],
          [                                                           'ALIKED+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked')],
          [                                                             'DISK+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='disk')],  
          [                                                                      'LoFTR', pipe_base.loftr_module(num_features=8000, upright=True)],        
          [                                                                  'DeDoDe v2', dedode2.dedode2_module(num_features=8000, upright=True)],                
        # [                                                 'SuperPoint+LightGlue (DIM)', superpoint_lightglue_module(nmax_keypoints=8000)],
        # [                                                     'ALIKED+LightGlue (DIM)', aliked_lightglue_module(nmax_keypoints=8000)],
        # [                                                       'DISK+LightGlue (DIM)', disk_lightglue_module(nmax_keypoints=8000)],
        # [                                                                'LoFTR (DIM)', loftr_module(nmax_keypoints=8000)],  
        ]
    
###

    pipe_renamed = []
    for pipe in pipes:
        new_name = pipe[0]
        old_name = pipe[1].get_id().replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
        pipe_renamed.append([old_name, new_name])

    for pipe in pipe_heads:
        new_name = pipe[0]
        old_name = pipe[1].get_id().replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
        pipe_renamed.append([old_name, new_name])

    bench_path = '../bench_data'
    save_to = 'res'
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
###

    # # not needed if launched run_bench.py before     
    # for b in benchmark_data.keys():
    #     b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
    
    csv_row_header = []
    csv_col_header = None
    csv_count = [[''] + [benchmark_data[b]['Name'].replace('IMC PhotoTourism','IMC-PT') for b in benchmark_data.keys()]]
    all_csv = []
    mask_row = []
    for ip in range(len(pipe_heads)):
        csv_list = []
        pipe_head = pipe_heads[ip][1]

        for b in benchmark_data.keys():
            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + pipe_head.get_id() + '_')
            to_save_file_suffix ='_' + benchmark_data[b]['name']

            if benchmark_data[b]['is_not_planar']:
                csv_list.append(to_save_file + 'fundamental_and_essential' + to_save_file_suffix + '.csv')
            else:
                csv_list.append(to_save_file + 'homography' + to_save_file_suffix + '.csv')
                
        fused_csv, _ = csv_merger(csv_list, include_match_count=True)
        if csv_col_header is None:
            csv_col_header = fused_csv[0]
                 
        v = pipe_head.get_id()
        for renamed in pipe_renamed:
            v = v.replace(renamed[0], renamed[1])
            v =v.replace('_outdoor_true','').replace('_outdoor_false','')
            
        csv_count.append([v] + [s[len('filtered_of_'):] for s in fused_csv[0] if ('filtered_of') in s])

        for i in range(1, len(fused_csv)):
            mask_row.append('sac' in fused_csv[i][0])
            all_csv.append(fused_csv[i][1:])
            csv_row_header.append(fused_csv[i][0])
   
    num_table = np.asarray(all_csv)
    mask_row = np.asarray(mask_row)
    
    # # with no filtered field
    # table_todo = [
    #     [    'MegaDepth',  [ 0,  1,  5,  9]],
    #     [      'ScanNet',  [10, 11, 15, 19]],
    #     [ 'PhotoTourism',  [26, 27, 31, 35, 39, 43]],        
    #     [    'Planar',  [20, 21, 25]],
    #     ['Non-planar',  [ 0,  1,  5,  9], [10, 11, 15, 19], [26, 27, 31, 39]],
    #     ]
    
    table_todo = [
        [    'MegaDepth',  [ 0,  1,  2,  6, 10]],
        [      'ScanNet',  [11, 12, 13, 17, 21]],
        [ 'PhotoTourism',  [29, 30, 31, 35, 39, 43, 47]],        
        [    'Planar',  [22, 23, 24, 28]],
        ['Non-planar',  [ 0,  1,  2,  6, 10], [11, 12, 13, 17, 21], [29, 30, 31, 35, 43]],
        ]
    
    for todo in table_todo:
        table_name = todo[0]
        idx_list = todo[1:]

        num_table_fa = np.zeros((0, len(idx_list[0])))
        num_table_fb = np.zeros((0, len(idx_list[0])))
        for idx in idx_list:        
            ccol = [csv_col_header[1:][i]  for i in idx]
    
            table = num_table[:, idx]
            num_table_a = num_table[mask_row][:, idx].astype(float)
            num_table_a = num_table_a[np.all(np.isfinite(num_table_a), axis=1)]
            num_table_fa = np.vstack((num_table_fa, num_table_a)) 

            num_table_b = num_table[~mask_row][:, idx].astype(float)
            num_table_b = num_table_b[np.all(np.isfinite(num_table_b), axis=1)]
            num_table_fb = np.vstack((num_table_fb, num_table_b)) 

        corr_table = np.triu(np.corrcoef(num_table_fa.transpose())) + np.tril(np.corrcoef(num_table_fb.transpose()), k=-1)
        corr_table = np.round(corr_table * 100) / 100
                
        for i, v in enumerate(ccol):
            if 'filtered' in v: v = 'Filt.'
            v = v.replace('pipeline', 'Pipeline')
            v = v.replace('F_precision', 'Prec.')
            v = v.replace('F_recall', 'Recall')
            v = v.replace('H_precision', 'Prec.')
            v = v.replace('H_recall', 'Recall')
            v = v.replace('F_AUC', 'AUC$^{F}$')
            v = v.replace('E_AUC', 'AUC$^{E}$')
            v = v.replace('H_AUC', 'AUC$^{H}$')
            v = v.replace('@5', '$_{\\text{@}5}$')
            v = v.replace('@10', '$_{\\text{@}10}$')
            v = v.replace('@15', '$_{\\text{@}15}$')
            v = v.replace('@20', '$_{\\text{@}20}$')
            v = v.replace('@(5,0.5)', '$_{\\text{@}(5,\\frac{1}{2})}$')
            v = v.replace('@(10,1)', '$_{\\text{@}(10,1)}$')
            v = v.replace('@(20,2)', '$_{\\text{@}(20,2)}$')
            v = v.replace('@avg_a', '$_\\measuredangle$')
            v = v.replace('@avg_m', '$_\\square$')
            v = v.replace('$$', '')
            ccol[i] = v
        
        os.makedirs(os.path.join(bench_path, save_to, 'latex'), exist_ok=True)
        latex_file = os.path.join(bench_path, save_to, 'latex', 'corr_' + table_name.lower() + '.tex')
        latex_table = to_latex_corr(table_name, ccol, corr_table)
        csv_write(latex_table, latex_file)
        compile_latex(latex_file)


    os.makedirs(os.path.join(bench_path, save_to, 'latex'), exist_ok=True)
    latex_file = os.path.join(bench_path, save_to, 'latex', 'match_count.tex')
    latex_table = to_latex_simple(csv_count, table_name='Average number of matches per image')
    csv_write(latex_table, latex_file)
    compile_latex(latex_file)
