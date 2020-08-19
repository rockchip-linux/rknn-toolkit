/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>

#include <VX/vx_khr_cnn.h>
#include <VX/vx_helper.h>
#include <VX/vx_ext_program.h>

/*-------------------------------------------
            Macros and Variables
-------------------------------------------*/

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

#define MAX_DIM_NUM 8

typedef struct _tensor_attr {
    vx_size     size[MAX_DIM_NUM];
    vx_size     stride_size[MAX_DIM_NUM];
    uint32_t    dim_num;
    vx_enum     dtype;
    uint32_t    total_bytes;
} tensor_attr_t;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static inline uint32_t get_type_bytes
    (
    const vx_enum type
    )
{
    switch( type )
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
    case VX_TYPE_FLOAT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
    case VX_TYPE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static uint32_t get_stride_size
    (
    vx_size    *size,
    uint32_t    dim_num,
    vx_enum     type,
    vx_size   * stride
    )
{
    uint32_t total_bytes;
    uint32_t i;

    if(NULL == size || NULL == stride) {
        return 0;
    }

    stride[0] = get_type_bytes( type );
    total_bytes = stride[0];
    for(i = 1; i < dim_num; i ++) {
        stride[i] = size[i - 1] * stride[i - 1];
        total_bytes *= size[i];
    }
    total_bytes *= size[0];
    return total_bytes;
}

static vx_status get_vx_tensor_attr
    (
    vx_tensor tensor,
    tensor_attr_t *attr
    )
{
    vx_status status;

    memset(attr, 0, sizeof(tensor_attr_t));

    // get num of dims
    status = vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &(attr->dim_num), sizeof(uint32_t));
    if (status != VX_SUCCESS) {
        printf("Error on vxQueryTensor VX_TENSOR_NUMBER_OF_DIMS status=%d\n", status);
        return status;
    }

    // get dims
    uint32_t size[MAX_DIM_NUM];
    memset(size, 0, MAX_DIM_NUM*sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(uint32_t) * (attr->dim_num));
    if (status != VX_SUCCESS) {
        printf("Error on vxQueryTensor VX_TENSOR_DIMS status=%d\n", status);
        return status;
    }
    for(int i = 0; i < attr->dim_num; i++) {
        attr->size[i] = size[i];
    }

    // get data type
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &(attr->dtype), sizeof(vx_enum));
    if (status != VX_SUCCESS) {
        printf("Error on vxQueryTensor VX_TENSOR_DATA_TYPE status=%d\n", status);
        return status;
    }

    // get stride size
    attr->total_bytes = get_stride_size(attr->size, attr->dim_num, attr->dtype, attr->stride_size);

    return VX_SUCCESS;
}

void __f32_to_f16(uint16_t* f16, float* f32, int num)
{
    float* src = f32;
    uint16_t* dst = f16;
    int i = 0;

    for (; i < num; i++) {
        float in = *src;

        uint32_t fp32 = *((uint32_t *) &in);
        uint32_t t1 = (fp32 & 0x80000000u) >> 16;  /* sign bit. */
        uint32_t t2 = (fp32 & 0x7F800000u) >> 13;  /* Exponent bits */
        uint32_t t3 = (fp32 & 0x007FE000u) >> 13;  /* Mantissa bits, no rounding */
        uint32_t fp16 = 0u;

        if( t2 >= 0x023c00u )
        {
            fp16 = t1 | 0x7BFF;     /* Don't round to infinity. */
        }
        else if( t2 <= 0x01c000u )
        {
            fp16 = t1;
        }
        else
        {
            t2 -= 0x01c000u;
            fp16 = t1 | t2 | t3;
        }

        *dst = (uint16_t) fp16;

        src ++;
        dst ++;
    }
}

void __f16_to_f32(float* f32, uint16_t* f16, int num)
{
    uint16_t* src = f16;
    float* dst = f32;
    int i = 0;

    for (; i < num; i++) {
        uint16_t in = *src;

        int32_t t1;
        int32_t t2;
        int32_t t3;
        float out;

        t1 = in & 0x7fff;         // Non-sign bits
        t2 = in & 0x8000;         // Sign bit
        t3 = in & 0x7c00;         // Exponent

        t1 <<= 13;                // Align mantissa on MSB
        t2 <<= 16;                // Shift sign bit into position

        t1 += 0x38000000;         // Adjust bias

        t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

        t1 |= t2;                 // Re-insert sign bit

        *((uint32_t*)&out) = t1;

        *dst = out;

        src ++;
        dst ++;
    }
}

/*-------------------------------------------
            CPU Kernel Functions
-------------------------------------------*/

static vx_status VX_CALLBACK cpu_kernel_function
    (
    vx_node node,
    const vx_reference* parameters,
    uint32_t paramNum
    )
{
    tensor_attr_t in_tensor_attr0;
    tensor_attr_t in_tensor_attr1;
    tensor_attr_t out_tensor_attr;

    int16_t * src_buffer0;
    int16_t * src_buffer1;
    int16_t * dst_buffer;

    vx_size view_start[MAX_DIM_NUM] = {0};

    vx_status status;

    // get input and output tensor attribute
    status = get_vx_tensor_attr((vx_tensor)parameters[0], &in_tensor_attr0);
    status = get_vx_tensor_attr((vx_tensor)parameters[1], &in_tensor_attr1);
    status = get_vx_tensor_attr((vx_tensor)parameters[2], &out_tensor_attr);

    // create input and output buffer
    src_buffer0 = (int16_t *)malloc(in_tensor_attr0.total_bytes);
    src_buffer1 = (int16_t *)malloc(in_tensor_attr1.total_bytes);
    dst_buffer = (int16_t *)malloc(out_tensor_attr.total_bytes);

    // read input data from input tensor
    status = vxCopyTensorPatch((vx_tensor)parameters[0], in_tensor_attr0.dim_num, view_start, in_tensor_attr0.size,
                                in_tensor_attr0.stride_size, src_buffer0, VX_READ_ONLY, 0);
    if(VX_SUCCESS != status) {
        printf("TruncateDiv Copy tensor 0 patch fail %d.", status);
        free(src_buffer0);
        free(src_buffer1);
        free(dst_buffer);
        return VX_FAILURE;
    }
    status = vxCopyTensorPatch((vx_tensor)parameters[1], in_tensor_attr1.dim_num, view_start, in_tensor_attr1.size,
                            in_tensor_attr1.stride_size, src_buffer1, VX_READ_ONLY, 0);
    if(VX_SUCCESS != status) {
        printf("TruncateDiv Copy tensor 1 patch fail %d.", status);
        free(src_buffer0);
        free(src_buffer1);
        free(dst_buffer);
        return VX_FAILURE;
    }

    uint32_t src_size = in_tensor_attr0.size[0];
    for (int i = 1; i < in_tensor_attr0.dim_num; i++) {
        src_size *= in_tensor_attr0.size[i];
    }
    uint32_t dst_size = out_tensor_attr.size[0];
    for (int i = 1; i < out_tensor_attr.dim_num; i++) {
        dst_size *= out_tensor_attr.size[i];
    }
    if (src_size != dst_size) {
        printf("Error, TruncateDiv op src_size(%d) != dst_size(%d)\n", src_size, dst_size);
    }

    float *tmp_buffer0 = (float *)malloc(src_size * sizeof(float));
    float *tmp_buffer1 = (float *)malloc(src_size * sizeof(float));

    __f16_to_f32(tmp_buffer0, src_buffer0, src_size);
    __f16_to_f32(tmp_buffer1, src_buffer1, src_size);

    for (int i = 0; i < src_size; i++) {
        tmp_buffer0[i] = tmp_buffer0[i] / tmp_buffer1[i];
    }

    __f32_to_f16(dst_buffer, tmp_buffer0, src_size);

    free(tmp_buffer0);
    free(tmp_buffer1);

    // save output data to output tensor
    status = vxCopyTensorPatch((vx_tensor)parameters[2], out_tensor_attr.dim_num, view_start, out_tensor_attr.size,
                                out_tensor_attr.stride_size, dst_buffer, VX_WRITE_ONLY, 0);
    if (VX_SUCCESS != status) {
        printf("Truncatediv Copy data to Tensor fail %d\n", status);
    }

    free(src_buffer0);
    free(src_buffer1);
    free(dst_buffer);

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK cpu_kernel_validator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK cpu_kernel_initializer
    (
    vx_node node,
    const vx_reference *parameters,
    uint32_t paramNum
    )
{
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK cpu_kernel_deinitializer
    (
    vx_node node,
    const vx_reference *parameters,
    uint32_t paramNum
    )
{
    return VX_SUCCESS;
}

/*-------------------------------------------
             VX Kernel Functions
-------------------------------------------*/

static vx_status VX_CALLBACK vx_kernel_validator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK vx_kernel_initializer
    (
    vx_node node,
    const vx_reference *parameters,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)

    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}   // globalWorkSize: image size in thread
    };

    vx_status status = VX_SUCCESS;

    // TODO: set vx kernel work size
    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = 64;
    shaderParam.globalWorkSize[1]   = 64;

    status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}

static vx_status VX_CALLBACK vx_kernel_deinitializer
    (
    vx_node node,
    const vx_reference *parameters,
    uint32_t paraNum
    )
{
    return VX_SUCCESS;
}

/*-------------------------------------------
              Kernel Declaration
-------------------------------------------*/

static vx_param_description_t kernel_params[] =
    {
        {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
        {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
        {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
    };

static vx_kernel_description_t cpu_kernel_info =
{
    0,
    "truncatediv_cpu",
    cpu_kernel_function,
    kernel_params,
    _cnt_of_array(kernel_params),
    cpu_kernel_validator,
    NULL,
    NULL,
    cpu_kernel_initializer,
    cpu_kernel_deinitializer
};

static vx_kernel_description_t vx_kernel_info =
{
    0,
    "truncatediv",
    NULL,
    kernel_params,
    _cnt_of_array(kernel_params),
    vx_kernel_validator,
    NULL,
    NULL,
    vx_kernel_initializer,
    vx_kernel_deinitializer
};

vx_kernel_description_t * vx_kernel_TruncateDiv_list[] =
{
    &cpu_kernel_info,
    &vx_kernel_info,
    NULL
};