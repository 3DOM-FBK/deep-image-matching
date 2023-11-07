from nets.aliked import *
from copy import deepcopy
from thop import profile
from thop.vision.calc_func import calculate_conv2d_flops, l_prod

def count_dcn(m, x, y):
    x = x[0]
    batch_size, in_planes, height, width = x.size()
    kernel_size = m.offset_conv.kernel_size[0]

    # offset_conv
    m.total_ops += calculate_conv2d_flops(input_size=list(x.shape),
                                          output_size=[batch_size, m.channel_num, height, width],
                                          kernel_size=list(m.offset_conv.weight.shape),
                                          groups=m.offset_conv.groups,
                                          bias=m.offset_conv.bias)

    # sample
    m.total_ops += 4*kernel_size*kernel_size*l_prod(x.shape)
    # deformable_conv
    m.total_ops += calculate_conv2d_flops(input_size=list(x.shape),
                                          output_size=list(y.shape),
                                          kernel_size=list(m.regular_conv.weight.shape),
                                          groups=m.regular_conv.groups,
                                          bias=m.regular_conv.bias)

def count_sddh(m, x, y):
    x, p = x[0], x[1]
    batch_size, dims, height, width = x.size()
    kernel_size = m.kernel_size
    num_pos = m.n_pos
    n_kpts = p[0].shape[0]

    # offset_conv
    m.total_ops += calculate_conv2d_flops(input_size=[batch_size * n_kpts, dims, kernel_size, kernel_size],
                                          output_size=[batch_size * n_kpts, 2 * num_pos, 1, 1],
                                          kernel_size=list(m.offset_conv[0].weight.shape),
                                          groups=m.offset_conv[0].groups,
                                          bias=m.offset_conv[0].bias)
    m.total_ops += calculate_conv2d_flops(input_size=[batch_size * n_kpts, 2 * num_pos, 1, 1],
                                          output_size=[batch_size * n_kpts, 2 * num_pos, 1, 1],
                                          kernel_size=list(m.offset_conv[2].weight.shape),
                                          groups=m.offset_conv[2].groups,
                                          bias=m.offset_conv[2].bias)

    # sample
    m.total_ops += 4*n_kpts*batch_size*dims*num_pos
    # deformable desc
    m.total_ops += calculate_conv2d_flops(input_size=[batch_size* n_kpts, dims, num_pos, 1],
                                          output_size=[batch_size* n_kpts, dims, num_pos, 1],
                                          kernel_size=list(m.sf_conv.weight.shape),
                                          groups=m.sf_conv.groups,
                                          bias=m.sf_conv.bias)

    m.total_ops += batch_size * n_kpts * dims * num_pos * dims
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") # avoid torch.meshgrid UserWarning for clean print
    device = 'cpu'
    N_kpts = 1000
    model_list = ['aliked-t16', 'aliked-n16', 'aliked-n32']
    
    for model_name in model_list:
        print(f'=============== Model={model_name} ===============')
        net = ALIKED(model_name, device=device, top_k=N_kpts, load_pretrained=False)
        image = torch.randn((1, 3, 480, 640), device=device)
        flops, params = profile(deepcopy(net),
                                inputs=(image,),
                                custom_ops={ DeformableConv2d: count_dcn,SDDH: count_sddh,},
                                verbose=False)
        print('{:<30}  {:<8} GFLops'.format('Computational complexity: ',  flops / 1e9))
        print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))