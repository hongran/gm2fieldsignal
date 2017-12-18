#include "dsp_gpu.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cassert>
#include <chrono>

// Complex scale
/*
static __device__ __host__ inline cufftDoubleComplex DoubleComplexScale(cufftDoubleComplex a, double s)
{
  cufftDoubleComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}*/

//Vector transform functors
struct ScaleShift
{
  double a;
  double b;
  ScaleShift(const double _a,const double _b) : a(_a) , b(_b) {}
  // tell CUDA that the following code can be executed on the CPU and the GPU
  __host__ __device__
    double operator()(const double & x) const {
      return a * x + b;
    }
};

struct Add
{
  Add(){}
  __host__ __device__ double operator()(double x, double y){
    return x + y;
  }
};

//Scale Add: A + k*B
struct ScaleAdd
{
  double k;
  ScaleAdd(const double _k) : k(_k){}
  __host__ __device__ double operator()(double x, double y){
    return x + k*y;
  }
};

struct Subtract
{
  Subtract(){}
  __host__ __device__ double operator()(double x, double y){
    return x - y;
  }
};

struct Square
{
  __host__ __device__ double operator()(double x){
    return x * x;
  }
};

struct AddSquare
{
  AddSquare(){}
  __host__ __device__ double operator()(double x, double y){
    return sqrt(x * x + y * y);
  }
};

struct ATan2
{
  ATan2(){}
  __host__ __device__ double operator()(double y, double x){
    return atan2(y,x);
  }
};

struct NormReal
{
  NormReal(){}
  __host__ __device__ double operator()(double x){
    return x*x;
  }
};

struct NormComplex
{
  NormComplex(){}
  __host__ __device__ double operator()(thrust::complex<double> x){
  //__host__ __device__ double operator()(dsp::cdouble x){
    return thrust::norm<double>(x);
  }
};

//Gpu kernels
__global__ void MakeMatrix(const double *d_data, double *d_A, size_t dimCol, size_t dimRow,size_t startIdx)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < dimCol && j< dimRow){
    d_A[i*dimRow + j] = pow(d_data[startIdx+j],i);
  }
}

__global__ void MakeCoefficientMatrix1(const double *d_A,double *d_B,size_t dimCol, size_t dimRow)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i < dimCol && j < dimCol && k< dimRow){
    d_B[i*dimCol*dimRow + j*dimRow + k] = d_A[k*dimCol+i] * d_A[k*dimCol+j];
  }
}

__global__ void MakeCoefficientMatrix2(const double *d_B,double *d_M,size_t dimCol, size_t dimRow)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  d_M[i*dimCol + j] = 0.0;
  if (i < dimCol && j < dimCol){
    for (size_t k = 0 ; k < dimRow ; k++){
      //Change to col-major
      d_M[j*dimCol + i] += d_B[i*dimCol*dimRow + j*dimRow +k];
    }
  }
}

__global__ void MakeRH1(const double *d_b, const double *d_A,double *d_C,size_t dimCol, size_t dimRow)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < dimCol && j < dimRow){
    d_C[j*dimCol + i] = d_A[j*dimCol + i] * d_b[j];
  }
}

__global__ void MakeRH2(const double *d_C,double *d_rh,size_t dimCol, size_t dimRow)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  d_rh[i] = 0.0;
  if (i < dimCol){
    for (size_t j = 0 ; j < dimRow ; j++){
      d_rh[i] += d_C[j*dimCol + i];
    }
  }
}

__global__ void ComplexScale(const double c,cufftDoubleComplex* a,size_t N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N){
    a[i].x = a[i].x * c;
    a[i].y = a[i].y * c;
  }
}
  
__global__ void FindUnwindingNumber(const double *D_phase,double *D_phase_jumps,size_t Length,size_t NBatch,double thresh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<NBatch){
    int k=0;
    D_phase_jumps[i*Length] = 0.0;
    for (unsigned int j = i*Length + 1; j < (i+1)*Length; j++) {
      double u = D_phase[j] - D_phase[j - 1];
      if (-u > thresh){
	k++;
      }
      if (u > thresh){
	k--;
      }
      D_phase_jumps[j]=k;
    }
  }
}

__global__ void BatchFindRange(const double * D_env,double * D_MaxAmp, unsigned int * D_iwf, unsigned int * D_fwf,size_t Length,size_t NBatch, double thresh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<NBatch){
    auto it = thrust::max_element(thrust::device,D_env+i*Length,D_env+(i+1)*Length);
    double max = *it;
    D_MaxAmp[i] = max;
    unsigned int k = thrust::distance(D_env+i*Length,it);
    D_iwf[i] = k;
    while (k<Length){
      if (D_env[i*Length+k]<max*thresh){
	D_fwf[i] = k;
	break;
      }
      k++;
    }
  }
}

__global__ void BatchFindFFTRange(const double * D_psd,unsigned int * D_max_idx_fft,unsigned int * D_i_fft,unsigned int * D_f_fft,size_t Length,size_t NBatch,unsigned int fft_peak_index_width)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<NBatch){
    auto it = thrust::max_element(thrust::device,D_psd+i*Length,D_psd+(i+1)*Length);
    unsigned int k = thrust::distance(D_psd+i*Length,it);
    D_max_idx_fft[i] = k;
    if(k<=fft_peak_index_width) D_i_fft[i]=1;
    else D_i_fft[i]=k-fft_peak_index_width;
    D_f_fft[i]=k+fft_peak_index_width;
    if(D_f_fft[i]>Length) D_f_fft[i]=Length;
  }
}

int linearSolverCHOL(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
  int bufferSize = 0;
  int *info = NULL;
  double *buffer = NULL;
  double *A = NULL;
  int h_info = 0;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize);

  cudaMalloc(&info, sizeof(int));
  cudaMalloc(&buffer, sizeof(double)*bufferSize);
  cudaMalloc(&A, sizeof(double)*lda*n);


  // prepare a copy of A because potrf will overwrite A with L
  cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
  cudaMemset(info, 0, sizeof(int));

  cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info);

  cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

  if ( 0 != h_info ){ 
    fprintf(stderr, "Error: Cholesky factorization failed\n");
  }

  cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice);

  cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info);

  cudaDeviceSynchronize();

  if (info  ) { cudaFree(info); }
  if (buffer) { cudaFree(buffer); }
  if (A     ) { cudaFree(A); }

  return 0;
}



namespace dsp {

//Vec
std::vector<double>& VecAdd(std::vector<double>& a, std::vector<double>& b)
{
  assert(a.size() == b.size());
  unsigned int N = a.size();
  thrust::device_vector<double> d_input1(N);
  thrust::device_vector<double> d_input2(N);
  thrust::device_vector<double> d_output(N);

  //transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>());
  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input1.begin());
  thrust::copy(b.begin(),b.end(),d_input2.begin());

  // Calculate the sum
  thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(),Add());

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

std::vector<double>& VecSubtract(std::vector<double>& a, std::vector<double>& b)
{
  assert(a.size() == b.size());

  unsigned int N = a.size();
  thrust::device_vector<double> d_input1(N);
  thrust::device_vector<double> d_input2(N);
  thrust::device_vector<double> d_output(N);

  //transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<double>());
  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input1.begin());
  thrust::copy(b.begin(),b.end(),d_input2.begin());

  // Calculate the difference
  thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(),Subtract());

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

std::vector<double>& VecScale(double c, std::vector<double>& a)
{

  unsigned int N = a.size();
  thrust::device_vector<double> d_input1(N);
  thrust::device_vector<double> d_output(N);
/*
  if (a.size()!=N){
    N = a.size();
    d_input1.resize(N,0.0);
    d_output.resize(N,0.0);
  }*/
  /*
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c * (*it);
  }*/

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input1.begin());

  // Calculate the modulo-ed phase
  thrust::transform(d_input1.begin(), d_input1.end(), d_output.begin(),ScaleShift(c,0.0));

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

std::vector<double>& VecShift(double c, std::vector<double>& a)
{
  unsigned int N = a.size();
  thrust::device_vector<double> d_input1(N);
  thrust::device_vector<double> d_output(N);
  /*if (a.size()!=N){
    N = a.size();
    d_input1.resize(N,0.0);
    d_output.resize(N,0.0);
  }*/
  /*
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c + (*it);
  }*/

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input1.begin());

  // Calculate the modulo-ed phase
  thrust::transform(d_input1.begin(), d_input1.end(), d_output.begin(),ScaleShift(1.0,c));

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

double Chi2(std::vector<double>& a){

  unsigned int N = a.size();
  thrust::device_vector<double> d_input1(N);
  thrust::device_vector<double> d_output(N);
  /*if (a.size()!=N){
    N = a.size();
    d_input1.resize(N,0.0);
  }*/

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input1.begin());

  return thrust::transform_reduce (d_input1.begin(),d_input1.end(),Square(),0.0,thrust::plus<double>());
}

std::vector<double> norm(const std::vector<double>& v)
{/*
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input(N);
  static thrust::device_vector<double> d_output(N);

  if (v.size()!=N){
    N = v.size();
    d_input.resize(N,0.0);
    d_output.resize(N,0.0);
  }*/
  // Allocate the memory
  std::vector<double> res;
  
  res.reserve(v.size());

  // Iterate and push back the norm.
  for (auto it = v.begin(); it < v.end(); ++it) {
    res.push_back(std::norm(*it));
  }
  
  //Let GPU do the norm of each element
  //copy to thrust device vectors
/*  thrust::copy(v.begin(),v.end(),d_input.begin());

  // calculate norm
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), NormReal());

  thrust::copy(d_output.begin(),d_output.end(),res.begin());
*/
  return res;
}

std::vector<double> norm(const std::vector<cdouble>& v)
{/*
  static unsigned int N = 1000;
  static thrust::device_vector<thrust::complex<double>> d_input(N);
  static thrust::device_vector<double> d_output(N);

  if (v.size()!=N){
    N = v.size();
    d_input.resize(N,0.0);
    d_output.resize(N,0.0);
  }*/
  // Allocate the memory
  std::vector<double> res;

  res.reserve(v.size());

  // Iterate and push back the norm.
  for (auto it = v.begin(); it < v.end(); ++it) {
    res.push_back(std::norm(*it));
  }

  //Let GPU do the norm of each element
  //copy to thrust device vectors
/*  static thrust::host_vector<thrust::complex<double>>h_input(v);
  thrust::copy(h_input.begin(),h_input.end(),d_input.begin());
  //thrust::copy(v.begin(),v.end(),d_input.begin());

  // calculate norm
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), NormComplex());

  thrust::copy(d_output.begin(),d_output.end(),res.begin());
*/
  return res;
}

//FFT

std::vector<cdouble> Processor::ifft(const std::vector<cdouble>& v,unsigned int N, unsigned int NBatch)
{
  // Grab some useful constants.
  assert(v.size() == N*NBatch);
  double Nroot = std::sqrt(N);

  thrust::device_vector<thrust::complex<double>> d_input_z(N*NBatch);
  thrust::device_vector<thrust::complex<double>> d_output_z(N*NBatch);

  if ( FFTSize != N || batch != NBatch){
    ResetPlan(N,batch);
  }

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<cdouble> wfm_vec(N*NBatch, cdouble(0.0, 0.0));
  const cufftDoubleComplex *h_data = reinterpret_cast<const cufftDoubleComplex *>(&v[0]);
  cufftDoubleComplex *h_data_res =  reinterpret_cast<cufftDoubleComplex *>(&wfm_vec[0]);

  cufftDoubleComplex *d_data = (cuDoubleComplex*)thrust::raw_pointer_cast(d_input_z.data());

  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleComplex) * N*NBatch, cudaMemcpyHostToDevice);
  cufftExecZ2Z(planZ2Z, d_data, d_data, CUFFT_INVERSE);
  cudaMemcpy(h_data_res, d_data, sizeof(cufftDoubleComplex) * N*NBatch, cudaMemcpyDeviceToHost);

  // cudafft is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
    *it /= Nroot;
  }

  return wfm_vec;
}

std::vector<cdouble> Processor::rfft(const std::vector<double> &v,unsigned int N, unsigned int NBatch)
{
  //Test Time cost
  // Grab some useful constants.
  assert(v.size() == N*NBatch);
  unsigned int n = N / 2 + 1;  // size of rfft
  double Nroot = std::sqrt(N);

  thrust::device_vector<double> d_input_d(N*NBatch);
  thrust::device_vector<thrust::complex<double>> d_output_z(n*NBatch);

  if ( FFTSize != N || batch != NBatch){
    ResetPlan(N,batch);
  }

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<cdouble> fft_vec(n, cdouble(0.0, 0.0));
  cufftDoubleReal *d_data = (cufftDoubleReal*)thrust::raw_pointer_cast(d_input_d.data());
  cufftDoubleComplex *d_data_res = (cuDoubleComplex*)thrust::raw_pointer_cast(d_output_z.data());
  const cufftDoubleReal *h_data = reinterpret_cast<const cufftDoubleReal *>(&v[0]);
  cufftDoubleComplex *h_data_res =  reinterpret_cast<cufftDoubleComplex *>(&fft_vec[0]);

  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleReal) * N*NBatch, cudaMemcpyHostToDevice);
  cufftExecD2Z(planD2Z, d_data, d_data_res);
  //renormalize
  dim3 DimBlock (16);
  dim3 DimGrid (n*NBatch/16+1);
  ComplexScale<<<DimGrid, DimBlock>>>(1/Nroot,d_data_res,n*NBatch);

  cudaMemcpy(h_data_res, d_data_res, sizeof(cufftDoubleComplex) * n*NBatch, cudaMemcpyDeviceToHost);

  /*for (auto it = fft_vec.begin(); it != fft_vec.end(); ++it) {
    *it /= Nroot;
  }*/

 /* auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
  double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
  std::cout << "Time = "<<t<<std::endl;
  */
  return fft_vec;
}

//std::vector<double> Processor::irfft(const std::vector<cdouble>& fft, bool is_odd,unsigned int n, unsigned int NBatch)
std::vector<double> Processor::irfft(const std::vector<cdouble>& fft, unsigned int N, unsigned int NBatch)
{
  // Grab some useful constants.
  unsigned int n = N / 2 + 1;  // size of rfft
  assert(fft.size() == n*NBatch);
  double Nroot = std::sqrt(N);

  thrust::device_vector<thrust::complex<double>> d_input_z(n*NBatch);
  thrust::device_vector<double> d_output_d(N*NBatch);

  if ( FFTSize != N || batch != NBatch){
    ResetPlan(N,batch);
  }

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<double> wfm_vec(N*NBatch, 0.0);
  cufftDoubleComplex *d_data = (cuDoubleComplex*)thrust::raw_pointer_cast(d_input_z.data());
  cufftDoubleReal *d_data_res = (cufftDoubleReal*)thrust::raw_pointer_cast(d_output_d.data());
  const cufftDoubleComplex *h_data = reinterpret_cast<const cufftDoubleComplex *>(&fft[0]);
  cufftDoubleReal *h_data_res =  reinterpret_cast<cufftDoubleReal *>(&wfm_vec[0]);

  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleComplex) * n*NBatch, cudaMemcpyHostToDevice);
  cufftExecZ2D(planZ2D, d_data, d_data_res);
  cudaMemcpy(h_data_res, d_data_res, sizeof(cufftDoubleReal) * N*NBatch, cudaMemcpyDeviceToHost);

  // cufft is unnormalized, so we need to fix that.
/*  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
  	*it /= Nroot;
  }*/
  VecScale(1/Nroot, wfm_vec);

  return wfm_vec;
}

std::vector<double> Processor::hilbert(const std::vector<double>& v ,unsigned int N, unsigned int NBatch)
{
  // Return the call to the fft version.
  auto fft_vec = rfft(v,N,NBatch);

  // Zero out the constant term.
  fft_vec[0] = cdouble(0.0, 0.0);

  // Multiply in the -i.
  for (auto it = fft_vec.begin() + 1; it != fft_vec.end(); ++it) {
    *it = cdouble((*it).imag(), -(*it).real());
  }


  // Reverse the fft.
  return irfft(fft_vec, N,NBatch);
}

std::vector<double> Processor::psd(const std::vector<cdouble>& v)
{
  return norm(v); 
}


//Window Filter

std::vector<cdouble> Processor::window_filter(const std::vector<cdouble>& spectrum, const std::vector<double>& freq, double low, double high,unsigned int N, unsigned int NBatch)
{
  //window_filter*******************************************
  //We could use thrust copy if necessary
  auto output = spectrum;
  auto interval = freq[1] - freq[0];

  unsigned int i_start = std::floor((low-freq[0])/interval);
  unsigned int i_end = std::floor((high-freq[0])/interval);
  //result vector 
  thrust::device_vector<thrust::complex<double>> d_spectrum;

  cdouble * h_data = reinterpret_cast<cdouble *>(&output[0]);
  cdouble * d_data = (cdouble *)thrust::raw_pointer_cast(d_spectrum.data());
  cudaMemcpy(d_data, h_data, sizeof(cdouble) * N*NBatch, cudaMemcpyHostToDevice);

  //filter, revise
  for(unsigned int j=0; j<NBatch; j++)
  {
    thrust::fill(d_spectrum.begin()+j*N,d_spectrum.begin()+j*N+i_start,thrust::complex<double>(0.0,0.0));
    thrust::fill(d_spectrum.begin()+j*N+i_end,d_spectrum.begin()+(j+1)*N,thrust::complex<double>(0.0,0.0));
  }
  cudaMemcpy(h_data, d_data, sizeof(cdouble) * N*NBatch, cudaMemcpyDeviceToHost);

  return output;
}

// Calculates the phase by assuming the real signal is harmonic.
std::vector<double> Processor::phase(const std::vector<double>& wf_re, const std::vector<double>& wf_im,unsigned int N, unsigned int NBatch)
{
  thrust::device_vector<double> d_wf_re(N*NBatch);
  thrust::device_vector<double> d_wf_im(N*NBatch);
  thrust::device_vector<double> d_phase(N*NBatch);

  // Set the phase vector
  std::vector<double> phase(wf_re.size(), 0.0);

 /*
  std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), phase.begin(),
                 [](double re, double im) { return std::atan2(im, re); });
*/	
   
  //Assign thrust vectors
  thrust::copy(wf_re.begin(),wf_re.end(),d_wf_re.begin());
  thrust::copy(wf_im.begin(),wf_im.end(),d_wf_im.begin());

  // Calculate the modulo-ed phase
  thrust::transform(d_wf_im.begin(), d_wf_im.end(), d_wf_re.begin(), d_phase.begin(),ATan2());

  thrust::copy(d_phase.begin(),d_phase.end(),phase.begin());

  // Now unwrap the phase
  //Get the winding numbers

  thrust::device_vector<double> d_phase_jumps(NBatch*N);
  double *D_phase = (double *)thrust::raw_pointer_cast(d_phase.data());
  double *D_phase_jumps = (double *)thrust::raw_pointer_cast(d_phase_jumps.data());
  dim3 DimBlock2 (16);
  dim3 DimGrid2 (NBatch/16+1);
  FindUnwindingNumber<<<DimGrid2, DimBlock2>>>(D_phase,D_phase_jumps,N,NBatch,kMaxPhaseJump);

  //Add the winding phases
  thrust::transform(d_phase.begin(), d_phase.end(), d_phase_jumps.begin(), d_phase.begin(),ScaleAdd(kTau));
  //Copy the unwinded phase back to host
  thrust::copy(d_phase.begin(),d_phase.end(),phase.begin());
  
  
  // Now unwrap the phase
 /* double thresh = 0.5 * kTau;
  double m = 0.0;
  bool gave_warning = false;

  int k = 0; // to track the winding number
  for (auto it = phase.begin() + 1; it != phase.end(); ++it) {
    
    // Add current total
    *it += k * kTau;
    m = *(it) - *(it - 1);
    
    // Check for large jumps, both positive and negative.
    while (std::abs(m) > kMaxPhaseJump) {
      
      if (-m > kMaxPhaseJump) {
        
        if ((m + kTau > thresh) && (!gave_warning)) {
          std::cout << "Warning: jump over threshold." << std::endl;
          gave_warning = true;
          break;
        }
        
        k++;
        *it += kTau;
        
      } else if (m > kMaxPhaseJump) {
        
        if ((m - kTau < -thresh) && (!gave_warning)) {
          std::cout << "Warning: jump over threshold." << std::endl;
          gave_warning = true;
          break;
        }
        
        k--;
        *it -= kTau;
      }

      m = *(it) - *(it - 1);
    }
  }*/
  
  return phase;
}

std::vector<double> Processor::envelope(const std::vector<double>& wf_re, const std::vector<double>& wf_im)
{
  unsigned int N = wf_re.size();
  thrust::device_vector<double> d_wf_re(N);
  thrust::device_vector<double> d_wf_im(N);
  thrust::device_vector<double> d_env(N);

  // Set the envelope function
  std::vector<double> env(N, 0.0);

 /* 	std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), env.begin(),
	[](double r, double i) { return std::sqrt(r*r + i*i); });
	*/
   
  //Assign thrust vectors
  thrust::copy(wf_re.begin(),wf_re.end(),d_wf_re.begin());
  thrust::copy(wf_im.begin(),wf_im.end(),d_wf_im.begin());

  thrust::transform(d_wf_re.begin(), d_wf_re.end(), d_wf_im.begin(), d_env.begin(),AddSquare());

  thrust::copy(d_env.begin(),d_env.end(),env.begin());

  return env;
}

//****************************************************************************8
void Processor::FindFitRange(unsigned int Length, unsigned int NBatch,double start_amplitude, double fft_peak_width,
    std::vector<double>& psd,std::vector<double>&env,double interval,
    std::vector<unsigned int>& iwf, std::vector<unsigned int>& fwf,
    std::vector<unsigned int>& max_idx_fft,std::vector<unsigned int>& i_fft,std::vector<unsigned int>& f_fft, 
    std::vector<double>& max_amp,std::vector<unsigned short>& health)
{
  max_amp.resize(NBatch);
  iwf.resize(NBatch);
  fwf.resize(NBatch);
  //initialize MaxAmp vector
  thrust::device_vector<double> d_MaxAmp(NBatch);
  thrust::device_vector<unsigned int> d_iwf(NBatch);
  thrust::device_vector<unsigned int> d_fwf(NBatch);
  thrust::device_vector<double> d_env(NBatch*Length);
  thrust::device_vector<double> d_psd(NBatch*Length);

  thrust::copy(env.begin(),env.end(),d_env.begin());
  thrust::copy(psd.begin(),psd.end(),d_psd.begin());

  double *D_env = (double *)thrust::raw_pointer_cast(d_env.data());
  double *D_MaxAmp = (double *)thrust::raw_pointer_cast(d_MaxAmp.data());
  unsigned int *D_iwf = (unsigned int *)thrust::raw_pointer_cast(d_iwf.data());
  unsigned int *D_fwf = (unsigned int *)thrust::raw_pointer_cast(d_fwf.data());
  dim3 DimBlock2 (16);
  dim3 DimGrid2 (NBatch/16+1);
  BatchFindRange<<<DimGrid2, DimBlock2>>>(D_env,D_MaxAmp,D_iwf,D_fwf,Length,NBatch,start_amplitude);
  thrust::copy(d_MaxAmp.begin(),d_MaxAmp.end(),max_amp.begin());
  thrust::copy(d_iwf.begin(),d_iwf.end(),iwf.begin());
  thrust::copy(d_fwf.begin(),d_fwf.end(),fwf.begin());

  for(unsigned int i=0; i<NBatch; i++)
  {
    if(iwf[i]>Length*0.95||iwf[i]>=fwf[i])
    {
      health[i]=0;
    }
    else
    {
      ;
    }
  }

  //max_idx_fft and i_fft , f_fft 
  max_idx_fft.resize(NBatch);
  i_fft.resize(NBatch);
  f_fft.resize(NBatch);
  unsigned int fft_peak_index_width = static_cast<int>(fft_peak_width/interval);
  unsigned int m;
  if (Length % 2 == 0) {
    m = Length/2+1;
  } else {
    m = (Length+1)/2;
  }

  thrust::device_vector<unsigned int> d_max_idx_fft(NBatch);
  thrust::device_vector<unsigned int> d_i_fft(NBatch);
  thrust::device_vector<unsigned int> d_f_fft(NBatch);

  double *D_psd = (double *)thrust::raw_pointer_cast(d_psd.data());
  unsigned int *D_max_idx_fft = (unsigned int *)thrust::raw_pointer_cast(d_max_idx_fft.data());
  unsigned int *D_i_fft = (unsigned int *)thrust::raw_pointer_cast(d_i_fft.data());
  unsigned int *D_f_fft = (unsigned int *)thrust::raw_pointer_cast(d_f_fft.data());
  BatchFindFFTRange<<<DimGrid2, DimBlock2>>>(D_psd,D_max_idx_fft,D_i_fft,D_f_fft,m,NBatch,fft_peak_index_width);
  thrust::copy(d_max_idx_fft.begin(),d_max_idx_fft.end(),max_idx_fft.begin());
  thrust::copy(d_i_fft.begin(),d_i_fft.end(),i_fft.begin());
  thrust::copy(d_f_fft.begin(),d_f_fft.end(),f_fft.begin());

}

// Helper function to get frequencies for FFT
std::vector<double> fftfreq(const std::vector<double>& tm, unsigned int NBatch) 
{
  int N = tm.size();
  double dt = (tm[N-1] - tm[0]) / (N - 1); // sampling rate

  return fftfreq(N, dt,NBatch);
}

std::vector<double> fftfreq(const int N, const double dt, unsigned int NBatch)
{
  // Instantiate return vector.
  std::vector<double> freq;

  // Handle both even and odd cases properly.
  unsigned int m; // size of fft_freq
  // Handle both even and odd cases properly.
  if (N % 2 == 0) {
    m = N/2+1;
    freq.resize(m);
    for(unsigned int j=0; j<NBatch; j++){ 
      for (unsigned int i = 0; i < m; i++) {
	freq[i+j*m] = i / (dt * N);
      }
    }
  } else {
    m = (N+1)/2;
    freq.resize(m);
    for(unsigned int j=0; j<NBatch; j++){ 
      for (unsigned int i = 0; i < m; i++) {
	freq[i+j*m] = i / (dt * N);
      }
    }
  }

  return freq;
}

int convolve(const std::vector<double>& v, const std::vector<double>& filter, std::vector<double>& res)
{
  int k = filter.size();
  int N = v.size();
  res.resize(N);

  // Check to make sure we can do something.
  if (N < k) {
    return -1;
  }

  // First take care of the beginning and end.
  for (int i = 0; i < k + 1; ++i) {
    res[i] = 0.0;
    res[N -1 - i] = 0.0;

    for (int j = i; j < i + k; ++j) {

      res[i] += v[abs(j - k/2)] * filter[j - i];
      res[N - 1 - i] += v[N - 1 - abs(k/2 - j)] * filter[j - i];
    }
  }

  // Now the rest of the elements.
  for (auto it = v.begin(); it != v.end() - k; ++it) {
    double val = std::inner_product(it, it + k, filter.begin(), 0.0);
    res[std::distance(v.begin(), it + k/2)] = val;
  }

  return 0;
}

std::vector<double> convolve(const std::vector<double>& v, const std::vector<double>& filter)
{
  std::vector<double> res(0.0, v.size());

  if (convolve(v, filter, res) == 0) {

    return res;

  } else {

    return v;
  }
}
 
int linear_fit(const std::vector<double>& x, const std::vector<double>& y, const std::vector<unsigned int> i_idx, const std::vector<unsigned int> f_idx , const size_t NPar,const unsigned int NBatch, const unsigned int Length,std::vector<std::vector<double>>& ParLists, std::vector<double>& Res)
{
  //Create the cuSolver
  static cusolverDnHandle_t handle = NULL;
  static cublasHandle_t cublasHandle = NULL; // used in residual evaluation
//  cudaStream_t stream = NULL;
  if (handle==NULL)cusolverDnCreate(&handle);
  if (cublasHandle==NULL)cublasCreate(&cublasHandle);
//  cudaStreamCreate(&stream);

  auto t0 = std::chrono::high_resolution_clock::now();
  assert(x.size() == y.size());
  assert(x.size() == NBatch*Length);
  for (unsigned int i=0;i<NBatch;i++){
    if (ParLists[i].size() != NPar) ParLists[i].resize(NPar);
  }

  Res = std::vector<double>(x.size(),0.0);
  //VecScale(0.0,Res);
  //minimize |Ax - b| 
  //const double * Vb = reinterpret_cast<const double *>(&y[i_idx]);
  //const double * Vdata = reinterpret_cast<const double *>(&x[i_idx]);
  //double * VSol = reinterpret_cast<double *>(&ParList[0]);
  //double * VRes = reinterpret_cast<double *>(&Res[i_idx]);
  const double * Vb = reinterpret_cast<const double *>(&y[0]);
  const double * Vdata = reinterpret_cast<const double *>(&x[0]);
  double * VRes = reinterpret_cast<double *>(&Res[0]);
//  double * h_A = nullptr;
//  double * h_M = nullptr;
  //Allocate Device memory
/*  double * d_A = nullptr;
  double * d_M = nullptr;
  double * d_b = nullptr;
  double * d_rh = nullptr;
  double * d_data =nullptr;
  double * d_par =nullptr;
*/
  static double * d_A = nullptr;
  static double * d_M = nullptr;
  static double * d_b = nullptr;
  static double * d_rh = nullptr;
  static double * d_data =nullptr;
  static double * d_res =nullptr;
  static double * d_par =nullptr;
  //size_t size_A = NPar*N_Eq*sizeof(double);
  size_t size_A = NPar*Length*sizeof(double);
  size_t size_M = NPar*NPar*sizeof(double);
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_M, size_M);
  //cudaMalloc(&d_b, N_Eq*sizeof(double));
  cudaMalloc(&d_b, NBatch*Length*sizeof(double));
  cudaMalloc(&d_rh, NPar*sizeof(double));
  //cudaMalloc(&d_data, N_Eq*sizeof(double));
  //cudaMalloc(&d_res, N_Eq*sizeof(double));
  cudaMalloc(&d_data, NBatch*Length*sizeof(double));
  cudaMalloc(&d_res, NBatch*Length*sizeof(double));
  cudaMalloc(&d_par, NPar*sizeof(double));
  /*
  if (d_A==nullptr) cudaMalloc(&d_A, size_A);
  if (d_M==nullptr) cudaMalloc(&d_M, size_M);
  if (d_b==nullptr) cudaMalloc(&d_b, N_Eq*sizeof(double));
  if (d_rh==nullptr) cudaMalloc(&d_rh, NPar*sizeof(double));
  if (d_data==nullptr) cudaMalloc(&d_data, N_Eq*sizeof(double));
  if (d_par==nullptr) cudaMalloc(&d_par, NPar*sizeof(double));
*/
  //copy to device
  cudaMemcpy(d_b, Vb, NBatch*Length*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, d_b, NBatch*Length*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_data, Vdata, NBatch*Length*sizeof(double), cudaMemcpyHostToDevice);

  for (unsigned int i=0;i<NBatch;i++){
    auto N_Eq = f_idx[i] - i_idx[i];
    double * VSol = reinterpret_cast<double *>(&ParLists[i][0]);
    //Make Matrix A
    dim3 DimBlock (NPar,16);
    dim3 DimGrid (1, N_Eq/16+1);
    MakeMatrix<<<DimGrid, DimBlock>>>(d_data,d_A,NPar,N_Eq,i*Length+i_idx[i]);
/*    auto t1 = std::chrono::high_resolution_clock::now();
    auto dtn1 = t1.time_since_epoch() - t0.time_since_epoch();
    double dt1 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn1).count();
    std::cout << "Time for making matrix  = "<<dt1<<std::endl;
*/
    const double alpha = 1.0;
    const double beta  = 0.0;

    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NPar, NPar, N_Eq, &alpha, d_A, N_Eq, d_A, N_Eq, &beta, d_M, NPar);
    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NPar, 1 , N_Eq, &alpha, d_A, N_Eq, d_b+i*Length+i_idx[i], N_Eq, &beta, d_rh, NPar);
    //  cusolverDnSetStream(handle, stream);
    //  cublasSetStream(cublasHandle, stream);

 /*   auto t2 = std::chrono::high_resolution_clock::now();
    auto dtn2 = t2.time_since_epoch() - t1.time_since_epoch();
    double dt2 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn2).count();
    std::cout << "Time for ATA  = "<<dt2<<std::endl;
*/
    linearSolverCHOL(handle, NPar, d_M, NPar, d_rh, d_par);

    //Calculate residual

    const double beta2  = -1.0;
    cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N_Eq, 1 , NPar, &alpha, d_A, N_Eq, d_par, NPar, &beta2, d_res+i*Length+i_idx[i], N_Eq);
    /*
       h_M = (double *)malloc(size_M);
       cudaMemcpy(h_M, d_M, size_M, cudaMemcpyDeviceToHost);
     */
    cudaMemcpy(VSol, d_par, NPar*sizeof(double), cudaMemcpyDeviceToHost);
  }
  //Copy residual vector as a whole
  cudaMemcpy(VRes, d_res, NBatch*Length*sizeof(double), cudaMemcpyDeviceToHost);

  auto t3 = std::chrono::high_resolution_clock::now();
  auto dtn3 = t3.time_since_epoch() - t0.time_since_epoch();
  double dt3 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn3).count();
  std::cout << "Time for solvingmatrix  = "<<dt3<<std::endl;

  /*for (int i=0;i<N_Eq;i++){
    std::cout << h_A[NPar*i+0] << " " << h_A[NPar*i+1] <<" "<<Vdata[i] <<std::endl;
  }
  std::cout<<NPar<<" "<<N_Eq<<std::endl;
  for (size_t i=0;i<NPar;i++){
    for (size_t j=0;j<NPar;j++){
      std::cout << h_M[NPar*i+j]<< " ";
    }
    std::cout << std::endl;
  }
  for (size_t j=0;j<NPar;j++){
    std::cout << VSol[j]<< std::endl;
  }
*/
  cudaFree(d_A);
  cudaFree(d_M);
  cudaFree(d_b);
  cudaFree(d_rh);
  cudaFree(d_data);
  cudaFree(d_res);
  cudaFree(d_par);

//  cudaFree(h_A);
  return 0;
}

struct im_harmonic
{
  im_harmonic(){}
  __device__ thrust::complex<double>  operator()(thrust::complex<double>x){
   thrust::complex<double> z;
   z.real(x.imag());
   z.imag(-x.real());
   return z;
  }
};

//structure of shift and square
struct varianceshiftop
{
  double mean;
  varianceshiftop(const double m) : mean(m) {}
  __device__ double operator()(double data) const 
  {
    return (data-mean)*(data-mean);
  }
};

//structure of compare
struct compare
{ 
  compare(){}
  __device__ double operator()(double x, double y) 
  {
    if(x>y)
    {
      return x;
    }
    else
    {
      return y;
    }
  }
};

//structure of Sqrt
struct Sqrt
{ 
  Sqrt(){}
  __device__ double operator()(double x) 
  {
    return sqrt(x);
  }
};
//structure of complex norm
struct my_complex_norm {
  __host__ __device__
  double operator()(thrust::complex<double> &d){
    return thrust::norm(d);
  }
};

//structure of absolute_value
template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};
/*
struct absolute_value
{
  __host__ __device__ double operator()(const double &x) const
  {
    return x < double(0) ? -x : x;
  }
};
*/
//IntegratedProcessor Functions
IntegratedProcessor::IntegratedProcessor(unsigned int BatchNum, unsigned int len):
  Length(len),NBatch(BatchNum)
{
    cufftPlan1d(&planD2Z, Length, CUFFT_D2Z, NBatch);
    cufftPlan1d(&planZ2D, Length, CUFFT_Z2D, NBatch);
    cufftPlan1d(&planZ2Z, Length, CUFFT_Z2Z, NBatch);
}

int IntegratedProcessor::SetFilters(double low, double high , double baseline_thresh, double peak_width)
{
  WindowFilterLow = low;
  WindowFilterHigh = high;
  Baseline_Freq_Thresh = baseline_thresh;
  fft_peak_width = peak_width;
  return 0;
}


int IntegratedProcessor::SetEdges(double ignore, double width)
{
  edge_ignore = ignore;
  edge_width = width;
  return 0;
}

int IntegratedProcessor::SetAmpThreshold(double Thresh)
{
  start_amplitude = Thresh;
  return 0;
}

void IntegratedProcessor::AnaSwith(bool sw)
{
  FreqAnaSwitch = sw;
}

void IntegratedProcessor::SetNFitPar(unsigned int N)
{
  NFitPar = N;
}

int IntegratedProcessor::Process(const std::vector<double>& wf,const std::vector<double>& tm, std::vector<double>& freq,
	std::vector<double>& filtered_wf, std::vector<double>& wf_im, std::vector<double>& baseline,
	std::vector<double>& psd, std::vector<double>& phi , std::vector<double>& env,
	std::vector<unsigned int>& iwf, std::vector<unsigned int>& fwf,
	std::vector<unsigned int> max_idx_fft,std::vector<unsigned int>& i_fft,std::vector<unsigned int>& f_fft, 
	std::vector<double>& max_amp,std::vector<unsigned short>& health,
	std::vector<std::vector<double>>& FreqResArray,std::vector<std::vector<double>>& FreqErrResArray,
	std::vector<std::vector<double>>& FitPars,std::vector<double>& ResidualOut)
{
  auto t0 = std::chrono::high_resolution_clock::now();
  //CalcFftFreq*********************************************
  double dt=tm[1]-tm[0];
  unsigned int m; // size of fft_freq
  // Handle both even and odd cases properly.
  if (Length % 2 == 0) {
    m = Length/2+1;
    //Resize freq to the same dimension
    freq.resize(m*NBatch);
    for(unsigned int j=0; j<NBatch; j++){ 
      for (unsigned int i = 0; i < m; i++) {
	freq[i+j*m] = i / (dt * Length);
      }
    }
  } else {
    m = (Length+1)/2;
    //Resize freq to the same dimension
    freq.resize(m*NBatch);
    for(unsigned int j=0; j<NBatch; j++){ 
      for (unsigned int i = 0; i < m; i++) {
	freq[i+j*m] = i / (dt * Length);
      }
    }
  }
  //rfft****************************************************
  //constant
  double Nroot=std::sqrt(Length);
  //CUDA FFT
  //load waveform to device

//  const cufftDoubleReal *h_data=reinterpret_cast<const cufftDoubleReal *>(&wf[0]);
  thrust::device_vector<double> d_fid(wf);
  //Initialize the result vector
  thrust::device_vector<thrust::complex<double>> d_fid_fft(m*NBatch);
  cufftDoubleReal *d_data = (cufftDoubleReal*)thrust::raw_pointer_cast(d_fid.data());
  cufftDoubleComplex *d_data_res = (cuDoubleComplex*)thrust::raw_pointer_cast(d_fid_fft.data());
  //Allocate memory
//  cudaMalloc((void **)&d_data, sizeof(cufftDoubleReal)*Length*NBatch);
//  cudaMalloc((void **)&d_data_res, sizeof(cufftDoubleComplex)*m*NBatch);
  //Copy data
//  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleReal)*Length*NBatch, cudaMemcpyHostToDevice);
  //Execute
  cufftExecD2Z(planD2Z, d_data, d_data_res);
  //Renormalize!! Maybe we could change the parameter!
  dim3 DimBlock (16); 
  dim3 DimGrid (m*NBatch/16+1);
  ComplexScale<<<DimGrid, DimBlock>>>(1/Nroot, d_data_res, m*NBatch);
  //window_filter*******************************************
  //We could use thrust copy if necessary
  auto interval=freq[1]-freq[0];
  unsigned int i_start = std::floor((WindowFilterLow-freq[0])/interval);
  unsigned int i_end = std::floor((WindowFilterHigh-freq[0])/interval);
  //result vector 
  thrust::device_vector<thrust::complex<double>> d_fid_fft_filtered=d_fid_fft;
  //filter, revise
  for(unsigned int j=0; j<NBatch; j++)
  {
    thrust::fill(d_fid_fft_filtered.begin()+j*m,d_fid_fft_filtered.begin()+j*m+i_start,thrust::complex<double>(0.0,0.0));
    /*for(unsigned int i=0; i<i_start;i++)
    {
      d_fid_fft_filtered[i+j*NBatch]=thrust::complex<double>(0.0,0.0);
    }*/
    thrust::fill(d_fid_fft_filtered.begin()+j*m+i_end,d_fid_fft_filtered.begin()+(j+1)*m,thrust::complex<double>(0.0,0.0));
    /*for(unsigned int i=i_end+1; i<m;i++)
    {
      d_fid_fft_filtered[i+j*NBatch]=thrust::complex<double>(0.0,0.0);
    }*/
  }
  //irfft***************************************************
  //Instantiate the result vector
  cufftDoubleReal *filtered_wf2;
  //Allocate memory
  cudaMalloc((void **)&filtered_wf2, sizeof(cufftDoubleReal) * Length * NBatch);
  cuDoubleComplex* V1 = (cuDoubleComplex*)thrust::raw_pointer_cast(d_fid_fft_filtered.data());
  //Execute
  cufftExecZ2D(planZ2D,V1, filtered_wf2);
  //save in vector d_filtered_wf
  thrust::device_vector<double> d_filtered_wf(&filtered_wf2[0],&filtered_wf2[Length*NBatch]);
  filtered_wf.resize(Length*NBatch);
  thrust::copy(d_filtered_wf.begin(),d_filtered_wf.end(),filtered_wf.begin());
  //imaginary harmonic complement***************************
  thrust::transform(d_fid_fft_filtered.begin(),d_fid_fft_filtered.end(),d_fid_fft_filtered.begin(),im_harmonic());
  //irfft***************************************************
  //Instantiate the result vector
  cufftDoubleReal *wf_im2;
  //Allocate memory
  cudaMalloc((void **)&wf_im2, sizeof(cufftDoubleReal) * Length * NBatch);
  cuDoubleComplex* V2 = (cuDoubleComplex*)thrust::raw_pointer_cast(d_fid_fft_filtered.data());
  //Execute
  cufftExecZ2D(planZ2D, V2, wf_im2);
  //save in vector wf_im
  thrust::device_vector<double> d_wf_im(&wf_im2[0],&wf_im2[Length*NBatch]);
  wf_im.resize(Length*NBatch);
  thrust::copy(d_wf_im.begin(),d_wf_im.end(),wf_im.begin());
  //window_filter to get baseline*******************************************
  //We could change it to parallel setting using thrust if necessary
  //We could put it in front of first filter to save time
  unsigned int i_end2 = std::floor((Baseline_Freq_Thresh-freq[0])/interval);
  thrust::device_vector<thrust::complex<double>> d_baseline_fft=d_fid_fft;
  //filter, revise
  //Cannot operate on device memory from host
  for(unsigned int j=0; j<NBatch; j++)
  {
    thrust::fill(d_baseline_fft.begin()+j*m,d_baseline_fft.begin()+j*m+i_end2+1,thrust::complex<double>(0.0,0.0));
   /* for(unsigned int i=i_end2+1; i<m;i++)
    {
      d_baseline_fft[i+j*NBatch]=thrust::complex<double>(0.0,0.0);
    }*/
  }

  //irfft*******************************************
  //Instantiate the result vector
  cufftDoubleReal *baseline2;
  //Allocate memory
  cudaMalloc((void **)&baseline2, sizeof(cufftDoubleReal) * Length * NBatch);
  cuDoubleComplex* V3 = (cuDoubleComplex*)thrust::raw_pointer_cast(d_baseline_fft.data());
  //Execute
  cufftExecZ2D(planZ2D,V3, baseline2);
  thrust::device_vector<double> d_baseline(&baseline2[0],&baseline2[Length*NBatch]); 
  //CalcPowerEnvAndPhase******************************
  thrust::device_vector<double> d_psd(NBatch*m); 
  //
  thrust::transform(d_fid_fft.begin(),d_fid_fft.end(),d_psd.begin(),my_complex_norm());
  //copy to host
  psd.resize(m*NBatch);
  thrust::copy(d_psd.begin(),d_psd.end(),psd.begin());
  //phase**********************************************
  thrust::device_vector<double> d_phase(NBatch*Length);
  // Calculate the modulo-ed phase
  thrust::transform(d_wf_im.begin(), d_wf_im.end(), d_filtered_wf.begin(),d_phase.begin(),ATan2());
  phi.resize(Length*NBatch);
  thrust::copy(d_phase.begin(),d_phase.end(),phi.begin());

  auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn1 = t1.time_since_epoch() - t0.time_since_epoch();
  double dt1 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn1).count();
  std::cout << "Time FFTs  = "<<dt1<<std::endl;

  // Now unwrap the phase
  //Get the winding numbers

  thrust::device_vector<double> d_phase_jumps(NBatch*Length);
  double *D_phase = (double *)thrust::raw_pointer_cast(d_phase.data());
  double *D_phase_jumps = (double *)thrust::raw_pointer_cast(d_phase_jumps.data());
  dim3 DimBlock2 (16);
  dim3 DimGrid2 (NBatch/16+1);
  FindUnwindingNumber<<<DimGrid2, DimBlock2>>>(D_phase,D_phase_jumps,Length,NBatch,kMaxPhaseJump);

  //Add the winding phases
  thrust::transform(d_phase.begin(), d_phase.end(), d_phase_jumps.begin(), d_phase.begin(),ScaleAdd(kTau));
  //Copy the unwinded phase back to host
  thrust::copy(d_phase.begin(),d_phase.end(),phi.begin());
  
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dtn2 = t2.time_since_epoch() - t1.time_since_epoch();
  double dt2 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn2).count();
  std::cout << "Time unrapping phase = "<<dt2<<std::endl;
  
  //envelope***************************************************  

  thrust::device_vector<double> d_env(NBatch*Length);

  thrust::transform(d_filtered_wf.begin(), d_filtered_wf.end(), d_wf_im.begin(), d_env.begin(),AddSquare());
  env.resize(Length*NBatch);
  thrust::copy(d_env.begin(),d_env.end(),env.begin());
  //BaselineCorrection
  //!!!!!!!!!Here I use filtered_wf to minux baseline
  // thrust::device_vector<double> d_wf_nobaseline(NBatch*Length);
  // thrust::transform(d_filtered_wf.begin(),d_filtered_wf.end(),d_baseline.begin(),d_wf_nobaseline.begin(),thrust::minus<double>());
  //CalcMaxAmp*****************************************************
  max_amp.resize(NBatch);
  iwf.resize(NBatch);
  fwf.resize(NBatch);
  //initialize MaxAmp vector
  thrust::device_vector<double> d_MaxAmp(NBatch);
  thrust::device_vector<unsigned int> d_iwf(NBatch);
  thrust::device_vector<unsigned int> d_fwf(NBatch);

  double *D_env = (double *)thrust::raw_pointer_cast(d_env.data());
  double *D_MaxAmp = (double *)thrust::raw_pointer_cast(d_MaxAmp.data());
  unsigned int *D_iwf = (unsigned int *)thrust::raw_pointer_cast(d_iwf.data());
  unsigned int *D_fwf = (unsigned int *)thrust::raw_pointer_cast(d_fwf.data());
  BatchFindRange<<<DimGrid2, DimBlock2>>>(D_env,D_MaxAmp,D_iwf,D_fwf,Length,NBatch,start_amplitude);
  thrust::copy(d_MaxAmp.begin(),d_MaxAmp.end(),max_amp.begin());
  thrust::copy(d_iwf.begin(),d_iwf.end(),iwf.begin());
  thrust::copy(d_fwf.begin(),d_fwf.end(),fwf.begin());

/*
  for(unsigned int i=0; i<NBatch; i++)
  {
    //thrust::device_vector<double>::iterator iter =  thrust::max_element(d_filtered_wf.begin()+i*Length,d_filtered_wf.begin()+(i+1)*Length);
    //iwf[i] = iter - (d_filtered_wf.begin()+i*Length);
    //max_amp[i] = *iter;

    auto iter = std::max_element(env.begin()+i*Length,env.begin()+(i+1)*Length);
    iwf[i]=std::distance(env.begin()+i*Length,iter);
    max_amp[i] = *iter;
    //d_MaxAmp[i]=thrust::transform_reduce(d_filtered_wf.begin(),d_filtered_wf.end(),absolute_value<double>(),0,thrust::maximum<double>());

    //    d_MaxAmp[i]=thrust::transform_reduce(d_filtered_wf.begin()+i*(Length),d_filtered_wf.begin()+(i+1)*Length-1,absolute_value<double>(),0,thrust::maximum<double>());
  }*/
  //thrust::copy(d_MaxAmp.begin(),d_MaxAmp.end(),max_amp.begin());
  
  //FindFidRange***************************************************************
  //Start from max amplitude, and find the first position env decays below amplitude*start_amplitude, assign that to fwf[i]
  /*
  std::vector<double> threshold(NBatch);
  std::transform(max_amp.begin(),max_amp.end(),threshold.begin(),std::bind1st(std::multiplies<double>(),start_amplitude));
  for(unsigned int i=0;i<NBatch;i++)
  {
    for(unsigned int j=iwf[i];j<Length;j++)
    {
      if(env[i*Length+j]<threshold[i])
      {
	fwf[i]=j;
	break;
      }
    }
  }*/
  /*for(unsigned int i=0;i<NBatch;i++)
  {
    for(unsigned int j=Length;j>0;j--)
    {
      if(filtered_wf[i*Length+j]<threshold[i])
      {
	fwf[i]=j;
	break;
      }
      else
      {
	;
      }
    }
  }*/
  for(unsigned int i=0; i<NBatch; i++)
  {
    if(iwf[i]>Length*0.95||iwf[i]>=fwf[i])
    {
      health[i]=0;
    }
    else
    {
      ;
    }
  }

  //max_idx_fft and i_fft , f_fft 
  max_idx_fft.resize(NBatch);
  i_fft.resize(NBatch);
  f_fft.resize(NBatch);
  unsigned int fft_peak_index_width = static_cast<int>(fft_peak_width/interval);

  thrust::device_vector<unsigned int> d_max_idx_fft(NBatch);
  thrust::device_vector<unsigned int> d_i_fft(NBatch);
  thrust::device_vector<unsigned int> d_f_fft(NBatch);

  double *D_psd = (double *)thrust::raw_pointer_cast(d_psd.data());
  unsigned int *D_max_idx_fft = (unsigned int *)thrust::raw_pointer_cast(d_max_idx_fft.data());
  unsigned int *D_i_fft = (unsigned int *)thrust::raw_pointer_cast(d_i_fft.data());
  unsigned int *D_f_fft = (unsigned int *)thrust::raw_pointer_cast(d_f_fft.data());
  BatchFindFFTRange<<<DimGrid2, DimBlock2>>>(D_psd,D_max_idx_fft,D_i_fft,D_f_fft,m,NBatch,fft_peak_index_width);
  thrust::copy(d_max_idx_fft.begin(),d_max_idx_fft.end(),max_idx_fft.begin());
  thrust::copy(d_i_fft.begin(),d_i_fft.end(),i_fft.begin());
  thrust::copy(d_f_fft.begin(),d_f_fft.end(),f_fft.begin());

  auto t3 = std::chrono::high_resolution_clock::now();
  auto dtn3 = t3.time_since_epoch() - t2.time_since_epoch();
  double dt3 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn3).count();
  std::cout << "Time for finding ranges = "<<dt3<<std::endl;

  //Fit phase to linear functions
  /*for(unsigned int j=0;j<NBatch;j++)
  {
    linear_fit(tm,phi, j*Length+iwf[j], j*Length+fwf[j] , NFitPar, FitPars[j], ResidualOut);
  }*/
  linear_fit(tm,phi, iwf, fwf , NFitPar,NBatch,Length, FitPars, ResidualOut);
  auto t4 = std::chrono::high_resolution_clock::now();
  auto dtn4 = t4.time_since_epoch() - t3.time_since_epoch();
  double dt4 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn4).count();
  std::cout << "Time for fit = "<<dt4<<std::endl;
  return 0;
}

//Common
std::vector<double> normalized_gradient(int npoints, int poln)
{
  std::vector<double> grad;

  // First get the spacing right.
  for (int i = 0; i < npoints; i++){
    grad.push_back(pow(i, poln));
  }

  // Subtract off the average.
  double avg = std::accumulate(grad.begin(), grad.end(), 0.0) / grad.size();

  for (uint i = 0; i < grad.size(); i++){
    grad[i] -= avg;
  }

  // Normalize by largest value.
  double max = *std::max_element(grad.begin(), grad.end());

  for (uint i = 0; i < grad.size(); i++){
    grad[i] /= max;
  }

  return grad;
}

// Add white noise to an array.

void addwhitenoise(std::vector<double>& v, double snr) {
  static std::default_random_engine gen(clock());
  std::normal_distribution<double> nrm;

  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  for (auto it = v.begin(); it != v.end(); ++it) {

    if (*it > max) {
      max = *it;
    } else {
      min = *it;
    }
    mean += *it;
  }

  // normalize to mean
  mean /= v.size();
  max -= mean;
  min = std::abs(min - mean);

  double amp = max > min ? max : min;
  double scale = amp / sqrt(snr);

  for (auto &x : v){
    x += nrm(gen) * scale;
  }
}

/*
arma::cx_mat wvd_cx(const std::vector<double>& v, bool upsample)
{
  int M, N;
  if (upsample) {

    M = 2 * v.size();
    N = v.size();

  } else {

    M = v.size();
    N = v.size();
  }

  // Initiate the return matrix
  arma::cx_mat res(M, N, arma::fill::zeros);

  // Artificially double the sampling rate by repeating each sample.
  std::vector<double> wf_re(M, 0.0);

  auto it1 = wf_re.begin();
  for (auto it2 = v.begin(); it2 != v.end(); ++it2) {
    *(it1++) = *it2;
    if (upsample) {
      *(it1++) = *it2;
    }
  }

  // Make the signal harmonic
  arma::cx_vec v2(M);
  arma::vec phase(M);

  auto wf_im = hilbert(wf_re);

  for (uint i = 0; i < M; ++i) {
    v2[i] = arma::cx_double(wf_re[i], wf_im[i]);
    phase[i] = (1.0 * i) / M * M_PI;
  }

  // Now compute the Wigner-Ville Distribution
  for (int idx = 0; idx < N; ++idx) {
    res.col(idx) = arma::fft(rconvolve(v2, idx));
  }

  return res;
}

arma::mat wvd(const std::vector<double>& v, bool upsample)
{
  int M, N;
  if (upsample) {

    M = 2 * v.size();
    N = v.size();

  } else {

    M = v.size();
    N = v.size();
  }

  // Instiate the return matrix
  arma::mat res(M, N, arma::fill::zeros);

  // Artificially double the sampling rate by repeating each sample.
  std::vector<double> wf_re(M, 0.0);

  auto it1 = wf_re.begin();
  for (auto it2 = v.begin(); it2 != v.end(); ++it2) {
    *(it1++) = *it2;
    if (upsample) {
      *(it1++) = *it2;
    }
  }

  // Make the signal harmonic
  arma::cx_vec v2(M);

  auto wf_im = hilbert(wf_re);

  for (int i = 0; i < M; ++i) {
    v2[i] = arma::cx_double(wf_re[i], wf_im[i]);
  }

  // Now compute the Wigner-Ville Distribution
  for (int idx = 0; idx < N; ++idx) {
    res.col(idx) = arma::real(arma::fft(rconvolve(v2, idx))) ;
  }

  return res;
}*/

std::vector<double> savgol3(const std::vector<double>& v)
{
  std::vector<double> res(0, v.size());
  std::vector<double> filter = {-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0};
  filter = VecScale((1.0 / 21.0) , filter);

  if (convolve(v, filter, res) == 0) {

    return res;

  } else {

    return v;
  }
}

std::vector<double> savgol5(const std::vector<double>& v)
{
  std::vector<double> res(0, v.size());
  std::vector<double> filter = {15.0, -55.0, 30.0, 135.0, 179.0, 135.0, 30.0, -55.0, 15.0};
  filter = VecScale((1.0 / 429.0) , filter);

  if (convolve(v, filter, res) == 0) {

    return res;

  } else {

    return v;
  }
}
} // ::dsp

