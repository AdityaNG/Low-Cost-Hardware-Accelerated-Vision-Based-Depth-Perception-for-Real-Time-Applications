#ifndef __ELAS_GPU_H__
#define __ELAS_GPU_H__

// Enable profiling
//#define PROFILE

#include <algorithm>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <stdint.h>
#include <functional>  

#include "elas.h"
#include "descriptor.h"
#include "triangle.h"
#include "matrix.h"


/**
 * Our ElasGPU class with all cuda implementations
 * Note where we extend the Elas class so we are calling
 * On all non-gpu functions there if they are not implemented
 */
class ElasGPU : public Elas {

public:

  // Constructor, input: parameters
  // Pass this to the super constructor
  ElasGPU(parameters param) : Elas(param) {}

// This was originally "private"
// Was converted to allow sub-classes to call this
// This assumes the user knows what they are doing
public:
  int32_t *d_u_vals, *d_v_vals;
  float *d_planes_a, *d_planes_b, *d_planes_c;
  bool *d_valids;
  int32_t *d_disparity_grid, *d_grid_dims;
  int32_t *d_P;
  float *d_D, *D_copy, *D_tmp;
  uint8_t *d_I1, *d_I2;

  void computeDisparity(std::vector<support_pt> p_support,std::vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                        uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D);

  void adaptiveMean (float* D);
  void cudaInit(int32_t size_total, int32_t* pixs_u, int32_t* pixs_v, int32_t disp_num, int32_t *grid_dims);
  void cudaDest();
};


#endif //__ELAS_GPU_H__
