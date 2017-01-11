import numpy as np
import theano
import theano.tensor as T

from theano.gpuarray.pool import GpuRoIPoolOp as ROIPoolingOp

op = ROIPoolingOp(pooled_h=2, pooled_w=2, spatial_scale=1.0)

t_data = T.ftensor4()
t_rois = T.fmatrix()

t_outs = op(t_data, t_rois)
t_c = t_outs[0].sum()

t_g_data = T.grad(t_c, t_data)[0]

f = theano.function([t_data, t_rois], t_outs + [t_g_data])
