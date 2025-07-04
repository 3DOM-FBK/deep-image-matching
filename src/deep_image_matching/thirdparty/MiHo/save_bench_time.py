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


def csv_merger(csv_list, extra_columns=0):

    if extra_columns == 0:
        avg_idx = [[ 3,  6, 'F_AUC@avg_a',  0], # MegaDepth
                   [ 6,  9, 'E_AUC@avg_a',  6],
                   [11, 14, 'F_AUC@avg_a',  9], # ScanNet
                   [14, 17, 'E_AUC@avg_a', 14],
                   [19, 22, 'H_AUC@avg_m', 19], # Planar
                   [24, 27, 'F_AUC@avg_a', 22], # PhotoTourism
                   [30, 33, 'E_AUC@avg_a', 30],
                   [27, 30, 'F_AUC@avg_m', 27],
                   [33, 36, 'E_AUC@avg_m', 33],
                   ]
    elif extra_columns == 1:           
        avg_idx = [[ 4,  7, 'F_AUC@avg_a',  0], # MegaDepth
                   [ 7, 10, 'E_AUC@avg_a',  7], 
                   [13, 16, 'F_AUC@avg_a', 10], # ScanNet
                   [16, 19, 'E_AUC@avg_a', 16],
                   [21, 25, 'H_AUC@avg_m', 19], # Planar
                   [28, 31, 'F_AUC@avg_a', 25], # PhotoTourism
                   [34, 37, 'E_AUC@avg_a', 34],
                   [31, 34, 'F_AUC@avg_m', 31],
                   [37, 40, 'E_AUC@avg_m', 37],
                   ]       
    else:   
        avg_idx = [[ 2,  2, '',             0], # MegaDepth
                   [ 5,  8, 'F_AUC@avg_a',  3], 
                   [ 8, 11, 'E_AUC@avg_a',  8],
                   [ 3,  3, '',             2],                   
                   [12, 12, '',            11], # ScanNet
                   [15, 18, 'F_AUC@avg_a', 13],
                   [18, 21, 'E_AUC@avg_a', 18],
                   [13, 13, '',            12],
                   [22, 22, '',            21], # Planar
                   [24, 28, 'H_AUC@avg_m', 23],
                   [23, 23, '',            22],
                   [29, 29, '',            28], # PhotoTourism
                   [32, 35, 'F_AUC@avg_a', 30],
                   [38, 41, 'E_AUC@avg_a', 38],
                   [35, 38, 'F_AUC@avg_m', 35],
                   [41, 44, 'E_AUC@avg_m', 41],
                   [30, 30, '',            29],
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
            l = trimmed_avg_idx[k][3]
                
            if k == len(trimmed_avg_idx) - 1:
                r = len(row_base)
            else:
                r = trimmed_avg_idx[k][1]
                
            if trimmed_avg_idx[k][2] != '':
                row_new =  row_base[l:r] + [str(row_avg.pop())] + row_new 
            else:
                row_avg.pop()
                row_new =  row_base[l:r] + row_new 
                               
        fused_csv.append(row_new)
        
    only_num_csv = [row[1:] for row in fused_csv[1:]]
    m = np.asarray(only_num_csv, dtype=float)
    sidx = np.argsort(-m, axis=0)
    sidx_ = np.argsort(sidx, axis=0)
    fused_csv_order = np.full((m.shape[0] + 1, m.shape[1] + 1), np.nan)
    fused_csv_order[1:,1:] = sidx_

    return fused_csv, fused_csv_order


def to_latex(csv_data, csv_order, renaming_list, header_hold=None, header_bar=None, prev_latex_table=None, add_footer=True, caption_string=None, page_align='landscape', remove_nan_column=False, resize_mode='width'):
    header_type = 'nmmmmmmmmmmmmssssssssssss'
    header_clr =  '-gbrtopvtopvlgbrtopvtopvl'
         
    pipe_count = csv_data[0][0]
    
    if header_hold is None:
        header_hold = header_type

    if header_bar is None:
        header_bar = header_clr

    use_ghost = True
    header_dict = {
        'n': '',
        'm': 'MegaDepth',
        's': 'ScanNet',
        'h': 'Planar',
        'p': 'IMC PhotoTourism'
        }
    
    bar_off = 0.05
    bar_dict = {
        '-': None,
        'b': 'blue',
        'r': 'red',
        't': 'teal',
        'o': 'orange',
        'p': 'purple',
        'v': 'violet',
        'l': 'olive',
        'g': 'CadetBlue',
        }
    bar_grad = np.asarray([ 0.5, 0.75, 0.875,   2  ])
    bar_grad_in =         ['70', '45', '35', '25']   
    bar_grad_out = '15'

    # removed unwanted rows
    if remove_nan_column == True:    
        csv_data_new = []
        csv_order_new = []
        
        for i in range(len(csv_data)):
            to_remove = False
            
            for j in csv_data[i]:
                if j == 'nan':
                    to_remove = True
                    break

            if to_remove == False:            
                csv_data_new.append(csv_data[i])
                csv_order_new.append(csv_order[i])
    
        csv_data = csv_data_new
        csv_order = csv_order_new
    
    # removed unwanted columns
    csv_data_new = []
    csv_order_new = []
    header_type_new = ''

    for i in range(len(csv_data)):
        csv_data_new.append([csv_data[i][j] for j in range(len(header_hold)) if header_hold[j] != '-'])
        csv_order_new.append([csv_order[i][j] for j in range(len(header_hold)) if header_hold[j] != '-'])

    for i in range(len(header_hold)):
        if header_hold[i] != '-':
            header_type_new = header_type_new + header_type[i]

    csv_data = csv_data_new
    csv_order = csv_order_new
    header_type = header_type_new
    header_bar = [header_bar[i] for i in range(len(header_hold)) if header_hold[i] != '-']    
    header_hold = header_hold.replace('-','') 
        
    # starting
    csv_head = csv_data[0]
    csv_head[0] = 'pipeline'
    
    csv_data = csv_data[1:]
    
    # adjusting ;;; in pipe column
    pipe_name = [row[0] for row in csv_data]
    base_index = -1
    base_count = -1
    for i, row in enumerate(pipe_name):
        bi = 0;
        for j in range(len(row)-1, -1, -1):
            if row[j] == ';': bi = bi + 1
            else: break
        if bi > base_count:
            base_count = bi
            base_index = i
                
    for i in range(base_count, 1, -1):    
        renaming_list.append([';' * i, ''])
    renaming_list.append([';', '+'])
    renaming_list.append(['_500', '_ 500'])
        
    pipe_renamed = []
    for pipe in pipe_name:
        for renamed in renaming_list:
            pipe = pipe.replace(renamed[0], renamed[1])
        if pipe[-1] == '+':
            pipe = pipe[:-1]
        pipe_renamed.append(pipe)
             
    print(pipe_renamed)   
        
    sort_idx = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pipe_renamed))]

    clean_pipe_renamed = []
    for i, pipe in enumerate(pipe_renamed):
        if i == base_index:
            clean_pipe_renamed.append(pipe)
        else:
            clean_pipe_renamed.append(pipe.replace(pipe_renamed[base_index], ''))

    print(clean_pipe_renamed)   
               
    clean_csv = [csv_head] + [[clean_pipe_renamed[i]] + csv_data[i][1:] for i in sort_idx]
    clean_csv_order = [csv_order[0]] + [csv_order[i + 1] for i in sort_idx]
        
    # csv_write([';'.join(csv_row) + '\n' for csv_row in clean_csv],'clean_table.csv')

    with_time = []
    for i, w in enumerate(clean_csv[0]):
        if 'runtime' in w: with_time.append([i, float(w[len('runtime_increment_from_'):-2])])
     
    np_data = np.zeros((len(clean_csv), len(clean_csv[0])))
    for i in range(1, len(clean_csv)):
        for j in range(len(clean_csv[0])):
            vv = clean_csv[i][j]
            
            try:
                v = float(vv)
            except:
                v = vv

            # numeric value            
            if isinstance(v, (int, float)):
                if np.isfinite(v):                
                    j_time = [qj for qj, qv in enumerate(with_time) if qv[0] == j]
                
                    if len(j_time) > 0:
                        # v in csv is the time increment pct wrt the base matching
                        v = "{n:6.3f}".format(n=(v+1)*with_time[j_time[0]][1])                
                    else:
                        v = "{n:6.2f}".format(n=v*100)

                    np_data[i, j] = v
    
                    # avoid alignement issues
                    if use_ghost == True:
                        for g in range(len(v)):
                            if v[g] != ' ':
                                break
                    v = "\hphantom{" + "0" * g + "}" + v[g:]
                else:
                    v = '\\hspace{0.5em}n/a'
                    np_data[i, j] = np.nan

                # highlight top pipelines for each column
                c_rank = int(clean_csv_order[i][j])        
                if c_rank < 3:
                    # color_rank = 'C' + str(c_rank)
                    color_rank = 'black'
                    v = '\\textcolor{' + color_rank + '}{\\contour{' + color_rank + '}{' + v + '}}'

            # text data in latex
            v = v.replace('miho_duplex_max_iter_ 500','\\textbf{MOP$_{0.5K}$+MiHo}')                  
            v = v.replace('miho_unduplex_max_iter_ 500','\\textbf{MOP$_{0.5K}$}')                  
            v = v.replace('miho_duplex_max_iter_1000','\\textbf{MOP$_{1.0K}$+MiHo}')                  
            v = v.replace('miho_unduplex_max_iter_1000','\\textbf{MOP$_{1.0K}$}')                  
            v = v.replace('miho_duplex_max_iter_1500','\\textbf{MOP$_{1.5K}$+MiHo}')                  
            v = v.replace('miho_unduplex_max_iter_1500','\\textbf{MOP$_{1.5K}$}')                  
            v = v.replace('miho_duplex_max_iter_2000','\\textbf{MOP$_{2.0K}$+MiHo}')                  
            v = v.replace('miho_unduplex_max_iter_2000','\\textbf{MOP$_{2.0K}$}')                  
            v = v.replace('NCC','\\textbf{NCC}')                  
            v = v.replace('0MAGSAC^','MAGSAC$_\\uparrow$')                  
            v = v.replace('0MAGSACv','MAGSAC$_\\downarrow$') 
            v = v.replace('AffNet+HardNet','$\\scriptsize\\substack{\\text{AffNet}\\\\\\text{HardNet}}$') 
                        
            clean_csv[i][j] = v

    # bar data
    v_min = np.nanmin(np_data[1:], axis=0)            
    v_max = np.nanmax(np_data[1:], axis=0)
    v_off = (v_max - v_min) * bar_off   
    v_min = np.maximum(v_min - v_off, 0)
    v_max = v_max + v_off
    val = (np_data[1:] - v_min) / (v_max - v_min)
    bar_val = np.full(np_data.shape, np.nan)
    bar_val[1:,1:] = val[:,1:]
    bar_val = np.round(bar_val * 1000) / 1000
    bar_vag = np.full(np_data.shape, np.nan, dtype=int)
    for i in range(bar_val.shape[0]):
        for j in range(bar_val.shape[1]):
            if np.isfinite(bar_val[i, j]):
                bar_vag[i, j] = np.sum(bar_val[i, j] < bar_grad) - 1

    # add bars
    bar_csv = []
    for i in range(len(clean_csv)):
        row = []
        for j in range(len(clean_csv[0])):
            if np.isfinite(bar_val[i, j]):
                row.append('\\Chart{' + clean_csv[i][j] + '}{' + str(bar_val[i, j]) + '}{' + bar_dict[header_bar[j]] + '}{' + bar_grad_in[bar_vag[i, j]] + '}{' + bar_grad_out + '}')
            elif (i > 0) and (j > 0):
                row.append('\\Chart{' + clean_csv[i][j] + '}{0.0}{' + bar_dict[header_bar[j]] + '}{' + bar_grad_in[0] + '}{' + bar_grad_out + '}')
            else:
                row.append(clean_csv[i][j])
        bar_csv.append(row)
    
    # add the & separator and the \\ at the end of the row
    latex_table = []
    for i, row in enumerate(bar_csv):
        if (i>=1) and ((i-1)%3!=2):
            midrule = ''
        else:
            midrule = '\t\t\t\t\\midrule\n'


        if (i==1): row[0] = '\\hspace{0.33em}' + row[0]                 
        if (i==2): row[0] = '\\hspace{0.66em}' + '+MAGSAC$_\\uparrow$'                 
        if (i==3): row[0] = '\\hspace{0.66em}' + '+MAGSAC$_\\downarrow$'             

        if (i > 3) and ((i-1)%3==0): row[0] = '\\hspace{0.66em}' + row[0]                 
        if (i > 3) and ((i-1)%3==1): row[0] = '\\hspace{1.3em}' + '+MAGSAC$_\\uparrow$'                 
        if (i > 3) and ((i-1)%3==2): row[0] = '\\hspace{1.3em}' + '+MAGSAC$_\\downarrow$'                 

        if (i>0) and (((i-1)//3)%2!=0): row[0] = '\\rowcolor{gray!15} ' + row[0]

        row[0] = '\t' * 4 + row[0]
        if i > 0: latex_table.append(' & '.join(row) + " \\\\\n" + midrule)
            
    if resize_mode == 'width':
        resize_what = '\t\t\\resizebox{\\textwidth}{!}{\n'
    else:
        resize_what = '\t\t\\resizebox*{!}{\\textheight}{\n'

    l = 0
    L = ''
    header_type_ = header_type + ' '
    for i in range(len(header_type_)):
        if i==0: continue
        if header_type_[i]!=header_type_[i-1]:
            if i!=1: L = L + ('L{\\MAX}' * (i-l-1)) + 'L{\\MAXX}'
            l = i

    header = [
        '\\documentclass[a4paper,' + page_align + ',10pt]{article}\n',
        '\\usepackage{fullpage}\n',        
        '\\usepackage{graphicx}\n',
        '\\usepackage{caption}\n',
        '\\captionsetup{labelformat=empty}\n',
        '\\usepackage{color}\n',
        '\\usepackage{adjustbox}\n',
        '\\usepackage{multirow}\n',
        '\\usepackage{booktabs}\n',
        '\\usepackage{amssymb}\n',
        '\\usepackage[table,usenames,dvipsnames]{xcolor}\n',
        '\\usepackage{amsmath}\n',
        '\\usepackage{multirow}\n',
        '\\usepackage{calc}\n',
        '\\usepackage{ulem}\n',
        '\\usepackage{nicefrac}\n',
        '\\usepackage[outline]{contour}\n',
        '\n',
        '\\newlength\\MAX\\setlength\\MAX{\\widthof{9999999999}}\n',
        '\\newlength\\MAXX\\setlength\\MAXX{\\widthof{99999999999}}\n',                
        '\\newcommand*\\Chart[5]{\\rlap{\\textcolor{#3!#5}{\\rule[-0.5ex]{\\MAX}{3ex}}}\\rlap{\\textcolor{#3!#4}{\\rule[-0.5ex]{#2\\MAX}{3ex}}}#1}\n',
        '\n',
        '\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\n',
        # colors from https://github.com/riccardosven/tableaucolors
        '\\definecolor{C0}{HTML}{1F77B4}\n',
        '\\definecolor{C1}{HTML}{FF7F0E}\n',
        '\\definecolor{C2}{HTML}{2CA02C}\n',
        '\\definecolor{C3}{HTML}{D62728}\n',
        '\\definecolor{C4}{HTML}{9467BD}\n',
        '\\definecolor{C5}{HTML}{8C564B}\n',
        '\\definecolor{C6}{HTML}{E377C2}\n',
        '\\definecolor{C7}{HTML}{7F7F7F}\n',
        '\\definecolor{C8}{HTML}{BCBD22}\n',
        '\\definecolor{C9}{HTML}{17BECF}\n',
        '\n',        
        '\\begin{document}\n',
        '\\pagestyle{empty}\n',
        '\t\\contourlength{0.1pt}\n',
        '\t\\contournumber{10}\n',
        '\t\\begin{table}[t!]\n',
        '\t\\renewcommand{\\arraystretch}{0}\n',
        '\t\\setlength{\\tabcolsep}{0pt}\n',
        '\t\\centering\n',
        resize_what,
        '\t\t\t\\begin{tabular}{L{\\widthof{+MOP+MiHo+NCC++++++}}' + L + '}\n',
    ]
    
    # header formatting
    l=0
    header_current = header_type[0]
    header_multi = []
    header_rule = []
    header_type_ = header_type + '$'
    for i in range(1,len(header_type_)):
        if header_type_[i] != header_current:
            header_multi.append('\\multicolumn{' + str(i-l)  + '}{c}{' + header_dict[header_current] + '}')
            if l + 1 != i: header_rule.append('\\cmidrule(lr){' +  str(l + 1)  + '-' + str(i) + '}')
            l = i
            header_current = header_type_[i]
        
    header.append('\t\t\t\t' + ' & '.join(header_multi) + ' \\\\\n')
    header.append('\t\t\t\t' + ''.join(header_rule) + '\n')
    
    header_spec = []               
    for v in csv_head:
        if 'filtered' in v: v = 'Filtered'
        if 'runtime' in v: v = 'Time (s)'
        v = v.replace('pipeline', 'Pipeline')
        v = v.replace('F_precision', 'Precision')
        v = v.replace('F_recall', 'Recall')
        v = v.replace('H_precision', 'Precision')
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
        header_spec.append(v)
        
    for i in range(len(header_spec)):
        if i > 0:
            header_spec[i] = '\\multicolumn{1}{c}{' + header_spec[i] + '}'
        
    header.append('\t\t\t\t' + ' & '.join(header_spec) + ' \\\\\n')    
    header.append('\t\t\t\t\\midrule\n')
    
    if caption_string is None:
        caption_string =  bar_csv[1][0][4:]
        
    footer = [
        '\t\t\t\end{tabular}\n',
        '\t\t}\n',
        '\t\t%\\caption{' + caption_string + '}\\label{none}\n',
        '\t\\end{table}\n',
        '\\end{document}\n',
    ]
    
    # can be set to concatenate tables
    if prev_latex_table is None:
        latex_table = header + latex_table
    else:
        latex_table = prev_latex_table + [header[-1]] + latex_table
        
    if add_footer:
        latex_table = latex_table + footer
            
    return latex_table


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


if __name__ == '__main__':    

    pipes = [
        [    '0MAGSAC^', pipe_base.magsac_module(px_th=1.00)],
        [    '0MAGSACv', pipe_base.magsac_module(px_th=0.75)],
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
          [ 'Key.Net+AffNet+HardNet+NNR', pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99)],
          [                   'SIFT+NNR', pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True)],     
          [       'SuperPoint+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint')],
          [           'ALIKED+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked')],
          [             'DISK+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='disk')],  
          [                      'LoFTR', pipe_base.loftr_module(num_features=8000, upright=True)],        
          [                  'DeDoDe v2', dedode2.dedode2_module(num_features=8000, upright=True)],                
      # # [ 'SuperPoint+LightGlue (DIM)', superpoint_lightglue_module(nmax_keypoints=8000)],
      # # [     'ALIKED+LightGlue (DIM)', aliked_lightglue_module(nmax_keypoints=8000)],
      # # [       'DISK+LightGlue (DIM)', disk_lightglue_module(nmax_keypoints=8000)],
      # # [                'LoFTR (DIM)', loftr_module(nmax_keypoints=8000)],  
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

    bench_path = '../bench_time'
    save_to = 'res'
    latex_path = 'latex'    
    latex_folder = os.path.join(bench_path, save_to, latex_path)
    os.makedirs(latex_folder, exist_ok=True)     
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
#           'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
#           'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
###

    header_hold = 'nmmm---m---mmsss---s---ss'
    header_bar =  '-gbrttttoooolgbrttttooool'
    full_el = 2
    
    latex_table_full = None
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
                
        fused_csv, fused_csv_order = csv_merger(csv_list, extra_columns=2)
        csv_write([';'.join(csv_row) + '\n' for csv_row in fused_csv], to_save_file.replace('_outdoor_true','').replace('_outdoor_false','')[:-1] + '.csv')

        if (ip % full_el == 0) and (ip != 0):
            csv_write(latex_table_full, save_to=os.path.join(latex_folder, 'all_' + str((ip // full_el) - 1)  + '.tex'))
            compile_latex(os.path.join(latex_folder, 'all_' + str((ip // full_el) - 1)  + '.tex'))
            latex_table_full = None

        latex_table_full = to_latex(fused_csv, fused_csv_order, pipe_renamed, prev_latex_table=latex_table_full, add_footer=((ip + 1) % full_el == 0) or (ip == len(pipe_heads) - 1), caption_string='Full results ' + str(ip // full_el), resize_mode='height')

        latex_table_full_standalone = to_latex(fused_csv, fused_csv_order, pipe_renamed, resize_mode='height')
        csv_write(latex_table_full_standalone, save_to=os.path.join(latex_folder, pipe_head.get_id() + '_full.tex'))
        compile_latex(os.path.join(latex_folder, pipe_head.get_id() + '_full.tex'))

        latex_table_standalone = to_latex(fused_csv, fused_csv_order, pipe_renamed, header_hold=header_hold, header_bar=header_bar, resize_mode='height')
        csv_write(latex_table_standalone, save_to=os.path.join(latex_folder, pipe_head.get_id() + '.tex'))
        compile_latex(os.path.join(latex_folder, pipe_head.get_id() + '.tex'))
    
    if not (latex_table_full is None):
        csv_write(latex_table_full, save_to=os.path.join(latex_folder, 'all_' + str(ip // full_el)  + '.tex'))
        compile_latex(os.path.join(latex_folder, 'all_' + str(ip // full_el)  + '.tex'))
