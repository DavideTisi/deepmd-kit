#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

REGISTER_OP("ProdHessianSeA")
.Attr("T: {float, double}")
.Input("net_deriv: T")
.Input("in_deriv: T")
.Input("net_hessian: T")
.Input("in_hessian: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("hessian: T")
.Output("atom_hessian: T");


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename Device, typename FPTYPE>
class ProdHessianSeAOp : public OpKernel {
 public:
  explicit ProdHessianSeAOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& net_hessian_tensor	= context->input(context_input_index++);
    const Tensor& in_hessian_tensor	= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    // TODO:shape of hessian
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));
    OP_REQUIRES (context, (0 == n_r_sel),					errors::InvalidArgument ("Rotational free only support all-angular information"));

    // Create an output tensor
    TensorShape hessian_shape ;
    hessian_shape.AddDim (nframes);
    hessian_shape.AddDim ( nall*nall*9);
    Tensor* hessian_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
						     hessian_shape, &hessian_tensor));
    // Create an output tensor
    TensorShape atom_hessian_shape ;
    atom_hessian_shape.AddDim (nframes);
    atom_hessian_shape.AddDim (nall * nall*nall * 9 );
    Tensor* atom_hessian_tensor = NULL;
    context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
						     atom_hessian_shape, &atom_hessian_tensor));
  //  // Create an output tensor
  //  TensorShape force_shape ;
  //  force_shape.AddDim (nframes);
  //  force_shape.AddDim (3 * nall);
  //  Tensor* force_tensor = NULL;
  //  int context_output_index = 0;
  //  OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	//					     force_shape, &force_tensor));
    
    // flat the tensors
    auto net_deriv = net_deriv_tensor.flat<FPTYPE>();
    auto in_deriv = in_deriv_tensor.flat<FPTYPE>();
    auto nlist = nlist_tensor.flat<int>();
    //auto force = force_tensor->flat<FPTYPE>();
    auto in_hessian = in_hessian_tensor.flat<FPTYPE>();
    auto net_hessian = net_hessian_tensor.flat<FPTYPE>();
    auto hessian = hessian_tensor->flat<FPTYPE>();
    auto atom_hessian = atom_hessian_tensor->flat<FPTYPE>();

    //assert (nframes == force_shape.dim_size(0));
    assert (nframes == hessian_shape.dim_size(0));
    assert (nframes == atom_hessian_shape.dim_size(0));
    assert (nframes == net_hessian_tensor.shape().dim_size(0));
    assert (nframes == in_hessian_tensor.shape().dim_size(0));
    assert (nframes == net_deriv_tensor.shape().dim_size(0));
    assert (nframes == in_deriv_tensor.shape().dim_size(0));
    assert (nframes == nlist_tensor.shape().dim_size(0));
    //assert (nall * 3 == force_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * ndescrpt == net_hessian_tensor.shape().dim_size(1)); // only nloc * ndescrpt are not zero only d^2e_k/dR_k^2
    assert (nloc * nloc * ndescrpt * 9 == in_hessian_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 4 == ndescrpt);	    
    
    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      int hessian_iter = kk * nall*nall*9 ;
      int atomic_hessian_iter = kk * nall * nall*nall*9 ;
      int net_hessian_iter = kk * nloc * ndescrpt ;
      int in_hessian_iter = kk * nloc * nloc * ndescrpt * 9 ;
      //int force_iter	= kk * nall * 3;
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int nlist_iter	= kk * nloc * nnei;

// TODO put the initialization in the main loop
      for (int ii = 0 ; ii < nall*nall*9 ; ++ii){
        int i_idx = ii ;
        hessian(hessian_iter + i_idx ) = 0. ;
        for (int jj = 0 ; jj<nall ; ++jj ){
          atom_hessian(atom_hessian_iter + i_idx + jj ) = 0. ;
        }
      }

      for (int ii = 0 ; ii < nall*nall*9 ; ++ii){
        int i_idx = ii ;
        for (int jj = 0; jj < nnei; ++jj){
          int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
          if (j_idx < 0) continue;
          int aa_start_1, aa_end_1;
	  // TODO the use of make_descript_range_2 and aa_1/aa_2 might be wrong  
          make_descript_range_1 (aa_start_1, aa_end_1, jj);
          for (int aa_1 = aa_start_1; aa_1 < aa_end_1; ++aa_1) {
            //FPTYPE de_dR1 =  net_deriv (net_iter + i_idx * ndescrpt + aa_1 ) ;
            //int dR_dr_iter =  in_iter * i_idx * ndescrpt + aa_1 ;
            for (int kk = 0; kk < nnei; ++kk){
              int z_idx = nlist (nlist_iter + i_idx * nnei + kk);
              if (z_idx < 0) continue;
              int aa_start_2, aa_end_2;
              make_descript_range_2 (aa_start_2, aa_end_2, kk );
              for (int aa_2 = aa_start_2 ; aa_2 < aa_end_2 ; ++aa_2 ) {
                //FPTYPE de_dR2 =  net_deriv (net_iter + i_idx * ndescrpt + aa_2 ) ;
                //int dR_dr_iter2 =  in_iter * i_idx * ndescrpt + aa_2 ;
                for (int dd0 = 0; dd0 < 3; ++dd0){
                  for (int dd1 = 0; dd1 < 3; ++dd1){
                    // compute d^2e_k/(dR_k)^2 * dR_k/dr_n dR_k/dr_m
                    FPTYPE tmp1 = net_hessian(net_hessian_iter+ i_idx * ndescrpt * ndescrpt + aa_1 * ndescrpt +aa_2  )*in_deriv (in_iter + i_idx * ndescrpt * 3 + aa_1 * 3 + dd0) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa_2 * 3 + dd1);
                    // compute de_k/dR_k * d^2R_k/(dr_n dr_m) 
                    FPTYPE tmp2 = net_deriv (net_iter + i_idx * ndescrpt + aa_1) *  in_hessian (in_hessian_iter + i_idx * nloc * ndescrpt+j_idx * ndescrpt + aa_1 );
                    atom_hessian(atomic_hessian_iter + i_idx * nnei * nnei * 9 + jj * nnei * 9 + kk * 9 + dd0 * 3 + dd1) += tmp1 + tmp2 ;
                    hessian (hessian_iter + jj * nnei * 9 + kk * 9 + dd0 * 3 + dd1) += tmp1 + tmp2  ;  
                  }
                }
              }
            }
          }
        }
      }


    }
  }
private:
  int n_r_sel, n_a_sel, n_a_shift;
  inline void
  make_descript_range (int & idx_start,
		       int & idx_end,
		       const int & nei_idx) {
    if (nei_idx < n_a_sel) {
      idx_start = nei_idx * 4;
      idx_end   = nei_idx * 4 + 4;
    }
    else {
      idx_start = n_a_shift + (nei_idx - n_a_sel);
      idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
  }
  inline void
  make_descript_range_1 (int & idx_start,
		       int & idx_end,
		       const int & nei_idx) {
    if (nei_idx < n_a_sel) {
      idx_start = nei_idx * 4;
      idx_end   = nei_idx * 4 + 4;
    }
    else {
      idx_start = n_a_shift + (nei_idx - n_a_sel);
      idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
  }

  inline void
  make_descript_range_2 (int & idx_start,
		       int & idx_end,
		       const int & nei_idx) {
    if (nei_idx < n_a_sel) {
      idx_start = nei_idx * 4;
      idx_end   = nei_idx * 4 + 4;
    }
    else {
      idx_start = n_a_shift + (nei_idx - n_a_sel);
      idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                  \
REGISTER_KERNEL_BUILDER(                                                                 \
    Name("ProdHessianSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdHessianSeAOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);

