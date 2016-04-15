from __future__ import absolute_import, print_function, division
import theano
import numpy
from theano import Op, Apply, config
from theano.tensor.extra_ops import CumsumOp

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .opt import register_opt as register_gpu_opt, op_lifter
from .type import GpuArrayType


class GpuCumsum(CumsumOp, Op):
    """
    Parameters
    ----------
    axis
        Can not be None. If you want the array flatten, do it before.
    """
    SUPPORTED_NDIMS = 3

    def __init__(self, axis):
        self.axis = axis
        # not sure if this should be here
        self.max_threads_dim0 = None
        self.max_grid_size1 = None
        self.max_gride_size2 = None


    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.axis)


    def c_code_cache_version(self):
        return (1,)


    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']


    def make_node(self, x): 
        if x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cumsum on 1D, 2D and\
                                       3D arrays are supported right now!')
        if self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={1}) out of bounds'.format(self.axis))

        x_ = as_gpuarray_variable(x, infer_context_name(x_))
        return Apply(self, [x_], [x_.type()])


    # ask Arnaud about this
    def make_thunk(self, node, storage_map, comput_map, no_recycling):
        pass


    # copied from neighbour.py
    def perform(self, node, inp, out, ctx):
        # Disable the perform method from the CPU version
        Op.perform(self, node, inp, out, ctx)


    def gpu_kernels(self, node, nodename):
        kernels = []
        # cumadd
        k_name = "k_cumadd"   
        k_var = "k_cumadd_" + nodename
        params =  
        dtype_x = node.inputs[0].dtype
        flags = Kernel.get_flags(dtype_x)
        code = """
        KERNEL void %(kname)s(float* input, float* output, size_t inputStrides[3],
                              size_t[3] outputStrides, int offsetY, int offsetZ, 
                              int beforeLastElementIdx, int lastElementIdx){
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            int dataOffsetY_input = idY * inputStrides[1] + idZ * inputStrides.[3];
            int dataOffsetY_output = idY * outputStrides[1] + idZ * outputStrides[2];
            int idx_last_input = lastElementIdx*inputStrides[0] + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides[0] + dataOffsetY_output;
            int idx_beforelast = beforeLastElementIdx*outputStrides[0] + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] + output[idx_beforelast];
            }
        """ % locals()
        kernels.append(Kernel(code=code, name="k_cumadd", params=params,
                              flags=flags, objvar=k_var))
        # finalCumSum
        k_name = "k_finalCumSum" 
        k_var = "k_finalCumSum_" + nodename
        params = 
        code = """
        void k_blockCumSum_%(nodename)s(float* input, float* output, int nbElementsPerCumsum, dim3 inputStrides, dim3 outputStrides, int offsetY, int offsetZ, float* blockSum) {
            // Regarding blockIdx and threadIdx, 'Cumsum' is always performed along the X axis.
            // The Y and Z axis of the grid will contain all independent cumsums of the 2D/3D case.

            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            // Check if current thread has data to process.
            if (globalThreadID >= ceil(nbElementsPerCumsum/2.0)) {
                return;
            }

            extern __shared__ float partialCumSum[];

            // Load data in shared memory
            k_fetchData_%(nodename)s(partialCumSum, input, globalThreadID, inputStrides, offsetY, offsetZ);

            // Use a dichotomy approach to compute the cumsum (i.e. balanced binary tree).
            // The tree is sweeped from the leaves to the root and from the root to the leaves.
            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            k_reductionPhase_%(nodename)s(partialCumSum);
            k_reversePhase_%(nodename)s(partialCumSum);

            // Write the final output to global memory
            k_pushData_%(nodename)s(partialCumSum, output, globalThreadID, outputStrides, offsetY, offsetZ);

            if (blockSum != NULL){
                if (threadIdx.x == blockDim.x - 1) {
                    blockSum[blockIdx.x*(gridDim.y*gridDim.z) + (blockIdx.y + offsetY)*gridDim.z + blockIdx.z + offsetZ] = partialCumSum[threadIdx.x*2 + 1];
                }
            }
        }

        int cumSum_%(nodename)s(CudaNdarray* input, CudaNdarray* output, int axis, int maxThreads, int maxGridY, int maxGridZ) {
            int shape[3] = { 1, 1, 1 };
            dim3 inputStrides(0, 0, 0);
            dim3 outputStrides(0, 0, 0);

            switch (CudaNdarray_NDIM(input))
            {
            case 1:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                break;
            case 2:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                shape[1] = CudaNdarray_HOST_DIMS(input)[1];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                inputStrides.y = CudaNdarray_HOST_STRIDES(input)[1];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                outputStrides.y = CudaNdarray_HOST_STRIDES(output)[1];
                break;
            case 3:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                shape[1] = CudaNdarray_HOST_DIMS(input)[1];
                shape[2] = CudaNdarray_HOST_DIMS(input)[2];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                inputStrides.y = CudaNdarray_HOST_STRIDES(input)[1];
                inputStrides.z = CudaNdarray_HOST_STRIDES(input)[2];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                outputStrides.y = CudaNdarray_HOST_STRIDES(output)[1];
                outputStrides.z = CudaNdarray_HOST_STRIDES(output)[2];
                break;
            default:
                return -1;
            }

            if (shape[axis] <= 1) {
                CudaNdarray_CopyFromCudaNdarray(output, input);
                return 0;
            }

            // Perform cumsum on array of even size.
            int nbElementsPerCumsum = shape[axis] - (shape[axis] %% 2);

            // Determine how many elements can be processed in one block.
            int dimBlockX = ceil( min(nbElementsPerCumsum, 2*maxThreads) / 2.0);

            // Determine how many blocks are needed in total.
            int dimGridX = ceil(nbElementsPerCumsum / (2.0*dimBlockX));  // Nb. of blocks needed per cumsum.
            int dimGridY;  // Nb. of independent cumsums (width).
            int dimGridZ;  // Nb. of independent cumsums (height).

            int tmp;
            switch (axis)
            {
            case 0:
                dimGridY = shape[1];
                dimGridZ = shape[2];
                break;
            case 1:
                dimGridY = shape[0];
                dimGridZ = shape[2];

                tmp = inputStrides.x;
                inputStrides.x = inputStrides.y;
                inputStrides.y = tmp;

                tmp = outputStrides.x;
                outputStrides.x = outputStrides.y;
                outputStrides.y = tmp;
                break;
            case 2:
                dimGridY = shape[1];
                dimGridZ = shape[0];

                tmp = inputStrides.x;
                inputStrides.x = inputStrides.z;
                inputStrides.z = tmp;

                tmp = outputStrides.x;
                outputStrides.x = outputStrides.z;
                outputStrides.z = tmp;
                break;
            default:
                return -1;
            }

            const int shapeBlockSum[2] = { dimGridX, dimGridY*dimGridZ };
            CudaNdarray* deviceBlockSum = (CudaNdarray*) CudaNdarray_NewDims(2, shapeBlockSum);

            // Perform `maxGridY`*`maxGridZ` cumsums in parallel.
            for (int offsetY = 0; offsetY < dimGridY; offsetY += maxGridY){
                int localDimGridY = min(dimGridY - offsetY, maxGridY);

                for (int offsetZ = 0; offsetZ < dimGridZ; offsetZ += maxGridZ){
                    int localDimGridZ = min(dimGridZ - offsetZ, maxGridZ);

                    dim3 dimGrid(dimGridX, localDimGridY, localDimGridZ);
                    dim3 dimBlock(dimBlockX, 1, 1);  // One cumsum per block.
                    int sharedBytes = (2*dimBlockX) * sizeof(float);

                    k_blockCumSum_%(nodename)s<<<dimGrid, dimBlock, sharedBytes>>>
                    (
                        CudaNdarray_DEV_DATA(input),
                        CudaNdarray_DEV_DATA(output),
                        nbElementsPerCumsum,
                        inputStrides,
                        outputStrides,
                        offsetY,
                        offsetZ,
                        CudaNdarray_DEV_DATA(deviceBlockSum)
                    );

                    if (dimGridX > 1) {
                        // Do a cumsum over the blockSum (recursive).
                        if (cumSum_%(nodename)s(deviceBlockSum, deviceBlockSum, 0, maxThreads, maxGridY, maxGridZ) == -1){
                            Py_DECREF(deviceBlockSum);
                            return -1;
                        }

                        // Since there are more than one block (i.e. `dimGridX > 1`)
                        //  report partial cumsums of previous blocks to subsequents ones.
                        dim3 dimGrid(dimGridX, localDimGridY, localDimGridZ);
                        dim3 dimBlock(dimBlockX, 1, 1);
                        k_finalCumSum_%(nodename)s<<<dimGrid, dimBlock>>>
                        (
                            CudaNdarray_DEV_DATA(output),
                            CudaNdarray_DEV_DATA(deviceBlockSum),
                            nbElementsPerCumsum,
                            outputStrides,
                            offsetY,
                            offsetZ
                        );
                    }

                    // If shape[axis] is odd, the last element is compute manually
                    if (shape[axis] != nbElementsPerCumsum){
                        dim3 dimGrid(1, localDimGridY, localDimGridZ);
                        dim3 dimBlock(1, 1, 1);
                        k_cumadd_%(nodename)s<<<dimGrid, dimBlock>>>
                        (
                            CudaNdarray_DEV_DATA(input),
                            CudaNdarray_DEV_DATA(output),
                            inputStrides,
                            outputStrides,
                            offsetY,
                            offsetZ,
                            shape[axis]-2,
                            shape[axis]-1
                        );
                    }
                }
            }
            Py_DECREF(deviceBlockSum);
            CNDA_THREAD_SYNC;
            return 0;
        }

        """       
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.context.kind != 'cuda':
            raise NotImplementedError("cuda only")
