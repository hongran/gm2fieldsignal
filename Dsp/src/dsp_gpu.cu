#include "dsp.h"
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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cassert>
#include <chrono>

typedef struct _cb_params{
  double scale;
} cb_params;

// Complex scale
static __device__ __host__ inline cufftDoubleComplex DoubleComplexScale(cufftDoubleComplex a, double s)
{
  cufftDoubleComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

//Callback routine for fft
static __device__ void ComplexPointwiseScale(void *a, size_t index, cufftDoubleComplex element , void *cb_info, void *sharedmem)
{
  cb_params * my_params = (cb_params *)cb_info;
  ((cufftDoubleComplex *)a)[index] = DoubleComplexScale(element,my_params->scale/4.0);
}

__device__ cufftCallbackStoreZ FFTCallbackPtr = ComplexPointwiseScale;

static __device__ void PointwiseScale(void *a, size_t index, cufftDoubleReal element , void *cb_info, void *sharedmem)
{
  cb_params * my_params = (cb_params *)cb_info;
  ((cufftDoubleReal *)a)[index] = element * my_params->scale / 4.0;
}

__device__ cufftCallbackStoreD Real_FFTCallbackPtr = PointwiseScale;

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
__global__ void MakeMatrix(const double *d_data, double *d_A, size_t dimCol, size_t dimRow)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < dimCol && j< dimRow){
    d_A[i*dimRow + j] = pow(d_data[j],i);
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

std::vector<double>& VecAdd(std::vector<double>& a, std::vector<double>& b)
{
  assert(a.size() == b.size());
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input1(N);
  static thrust::device_vector<double> d_input2(N);
  static thrust::device_vector<double> d_output(N);

  if (a.size()!=N){
    N = a.size();
    d_input1.resize(N,0.0);
    d_input2.resize(N,0.0);
    d_output.resize(N,0.0);
  }

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
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input1(N);
  static thrust::device_vector<double> d_input2(N);
  static thrust::device_vector<double> d_output(N);

  if (a.size()!=N){
    N = a.size();
    d_input1.resize(N,0.0);
    d_input2.resize(N,0.0);
    d_output.resize(N,0.0);
  }

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
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input(N);
  static thrust::device_vector<double> d_output(N);

  if (a.size()!=N){
    N = a.size();
    d_input.resize(N,0.0);
    d_output.resize(N,0.0);
  }
  /*
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c * (*it);
  }*/

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input.begin());

  // Calculate the modulo-ed phase
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),ScaleShift(c,0.0));

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

std::vector<double>& VecShift(double c, std::vector<double>& a)
{
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input(N);
  static thrust::device_vector<double> d_output(N);

  if (a.size()!=N){
    N = a.size();
    d_input.resize(N,0.0);
    d_output.resize(N,0.0);
  }
  /*
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c + (*it);
  }*/

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input.begin());

  // Calculate the modulo-ed phase
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),ScaleShift(1.0,c));

  thrust::copy(d_output.begin(),d_output.end(),a.begin());
  return a;
}

double VecChi2(std::vector<double>& a){
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_input(N);

  if (a.size()!=N){
    N = a.size();
    d_input.resize(N,0.0);
  }

  //Assign thrust vectors
  thrust::copy(a.begin(),a.end(),d_input.begin());

  return thrust::transform_reduce (d_input.begin(),d_input.end(),Square(),0.0,thrust::plus<double>());
}

//FFT

std::vector<cdouble> ifft(const std::vector<cdouble>& v)
{
  static cufftHandle plan;
  static int FFTSize = 0;
  // Grab some useful constants.
  int N = v.size();
  double Nroot = std::sqrt(N);

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<cdouble> wfm_vec(N, cdouble(0.0, 0.0));
  const int batch = 1;
  cufftDoubleComplex *d_data;
  const cufftDoubleComplex *h_data = reinterpret_cast<const cufftDoubleComplex *>(&v[0]);
  cufftDoubleComplex *h_data_res =  reinterpret_cast<cufftDoubleComplex *>(&wfm_vec[0]);

  if (FFTSize != N){
    // Define a structure used to pass in the device address of the scale factor
    cb_params h_params;

    h_params.scale = Nroot;

    // Allocate device memory for parameters
    cb_params *d_params;
    cudaMalloc((void **)&d_params, sizeof(cb_params));

    // Copy host memory to device
    cudaMemcpy(d_params, &h_params, sizeof(cb_params),cudaMemcpyHostToDevice);

    // The host needs to get a copy of the device pointer to the callback
    cufftCallbackStoreZ hostCopyOfCallbackPtr;

    cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr,FFTCallbackPtr,sizeof(hostCopyOfCallbackPtr));

    cufftPlan1d(&plan, N, CUFFT_Z2Z, batch);

    //Set Callback
//    cufftXtSetCallback(plan,(void **)&hostCopyOfCallbackPtr,CUFFT_CB_ST_COMPLEX_DOUBLE,(void **)&d_params);

    FFTSize = N;
  }

  cudaMalloc((void **)&d_data, sizeof(cufftDoubleComplex) * N);

  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE);
  cudaMemcpy(h_data_res, d_data, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost);

  cudaFree(d_data);

  // cudafft is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
    *it /= Nroot;
  }

  return wfm_vec;
}

std::vector<cdouble> rfft(const std::vector<double> &v)
{
  static cufftHandle plan;
  static int FFTSize = 0;
  //Test Time cost
//  auto t0 = std::chrono::high_resolution_clock::now();
  // Grab some useful constants.
  int N = v.size();  
  int n = N / 2 + 1;  // size of rfft
  double Nroot = std::sqrt(N);

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<cdouble> fft_vec(n, cdouble(0.0, 0.0));
  const int batch = 1;
  cufftDoubleReal *d_data;
  cufftDoubleComplex *d_data_res;
  const cufftDoubleReal *h_data = reinterpret_cast<const cufftDoubleReal *>(&v[0]);
  cufftDoubleComplex *h_data_res =  reinterpret_cast<cufftDoubleComplex *>(&fft_vec[0]);

  if (FFTSize != N){
    // Define a structure used to pass in the device address of the scale factor
    cb_params h_params;

    h_params.scale = Nroot;

    // Allocate device memory for parameters
    cb_params *d_params;
    cudaMalloc((void **)&d_params, sizeof(cb_params));

    // Copy host memory to device
    cudaMemcpy(d_params, &h_params, sizeof(cb_params),cudaMemcpyHostToDevice);

    // The host needs to get a copy of the device pointer to the callback
    cufftCallbackStoreZ hostCopyOfCallbackPtr;

    cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr,FFTCallbackPtr,sizeof(hostCopyOfCallbackPtr));

    cufftPlan1d(&plan, N, CUFFT_D2Z, batch);
    //Set Callback
//    cufftXtSetCallback(plan,(void **)&hostCopyOfCallbackPtr,CUFFT_CB_ST_COMPLEX_DOUBLE,(void **)&d_params);

    FFTSize = N;
  }

  cudaMalloc((void **)&d_data, sizeof(cufftDoubleReal) * N);
  cudaMalloc((void **)&d_data_res, sizeof(cufftDoubleComplex) * n);

  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleReal) * N, cudaMemcpyHostToDevice);
  cufftExecD2Z(plan, d_data, d_data_res);
  //renormalize
  dim3 DimBlock (16);
  dim3 DimGrid (N/16+1);
  ComplexScale<<<DimGrid, DimBlock>>>(1/Nroot,d_data_res,N);

  cudaMemcpy(h_data_res, d_data_res, sizeof(cufftDoubleComplex) * n, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_data_res);

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


std::vector<double> irfft(const std::vector<cdouble>& fft, bool is_odd)
{
  static cufftHandle plan;
  static int FFTSize = 0;
  // Grab some useful constants.
  int n = fft.size();
  int N = 2 * (n - 1) + 1 * is_odd;
  double Nroot = std::sqrt(N);

  //CUDA FFT
  // Instantiate the result vector.
  std::vector<double> wfm_vec(N, 0.0);
  const int batch = 1;
  cufftDoubleComplex *d_data;
  cufftDoubleReal *d_data_res;
  const cufftDoubleComplex *h_data = reinterpret_cast<const cufftDoubleComplex *>(&fft[0]);
  cufftDoubleReal *h_data_res =  reinterpret_cast<cufftDoubleReal *>(&wfm_vec[0]);

  if (FFTSize != N){
    // Define a structure used to pass in the device address of the scale factor
    cb_params h_params;

    h_params.scale = Nroot;

    // Allocate device memory for parameters
    cb_params *d_params;
    cudaMalloc((void **)&d_params, sizeof(cb_params));

    // Copy host memory to device
    cudaMemcpy(d_params, &h_params, sizeof(cb_params),cudaMemcpyHostToDevice);

    // The host needs to get a copy of the device pointer to the callback
    cufftCallbackStoreZ hostCopyOfCallbackPtr;

    cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr,Real_FFTCallbackPtr,sizeof(hostCopyOfCallbackPtr));

    cufftPlan1d(&plan, N, CUFFT_Z2D, batch);
    //Set Callback
//    cufftXtSetCallback(plan,(void **)&hostCopyOfCallbackPtr,CUFFT_CB_ST_REAL_DOUBLE,(void **)&d_params);

    FFTSize = N;
  }

  cudaMalloc((void **)&d_data, sizeof(cufftDoubleComplex) * n);
  cudaMalloc((void **)&d_data_res, sizeof(cufftDoubleReal) * N);
  
  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleComplex) * n, cudaMemcpyHostToDevice);
  cufftExecZ2D(plan, d_data, d_data_res);
  cudaMemcpy(h_data_res, d_data_res, sizeof(cufftDoubleReal) * N, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_data_res);
  // cufft is unnormalized, so we need to fix that.
/*  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
  	*it /= Nroot;
  }*/
  VecScale(1/Nroot, wfm_vec);

  return wfm_vec;
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

//Window Filter

std::vector<cdouble> window_filter(const std::vector<cdouble>& spectrum, const std::vector<double>& freq, double low, double high)
{
  auto output = spectrum;
  auto interval = freq[1] - freq[0];
  unsigned int i_start = std::floor((low - freq[0])/interval);
  unsigned int i_end = std::floor((high - freq[0])/interval);
  for (unsigned int i=0 ; i<spectrum.size() ; i++){
    if (i<i_start || i>i_end){
      output[i] = cdouble(0.0, 0.0);
    }
  }
  return output;
}

// Calculates the phase by assuming the real signal is harmonic.
std::vector<double> phase(const std::vector<double>& v)
{
	return phase(v, hilbert(v));
}

std::vector<double> phase(const std::vector<double>& wf_re, 
                               const std::vector<double>& wf_im)
{
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_wf_re(N);
  static thrust::device_vector<double> d_wf_im(N);
  static thrust::device_vector<double> d_phase(N);

  if (wf_re.size()!=N){
    N = wf_re.size();
    d_wf_re.resize(N,0.0);
    d_wf_im.resize(N,0.0);
    d_phase.resize(N,0.0);
  }
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
  double thresh = 0.5 * kTau;
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
  }
  
  return phase;
}

std::vector<double> envelope(const std::vector<double>& v)
{
	return envelope(v, hilbert(v));
}

std::vector<double> envelope(const std::vector<double>& wf_re, const std::vector<double>& wf_im)
{
  static unsigned int N = 1000;
  static thrust::device_vector<double> d_wf_re(N);
  static thrust::device_vector<double> d_wf_im(N);
  static thrust::device_vector<double> d_env(N);

  if (wf_re.size()!=N){
    N = wf_re.size();
    d_wf_re.resize(N,0.0);
    d_wf_im.resize(N,0.0);
    d_env.resize(N,0.0);
  }
  // Set the envelope function
  std::vector<double> env(wf_re.size(), 0.0);

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
 
int linear_fit(const std::vector<double>& x, const std::vector<double>& y, const unsigned int i_idx, const unsigned int f_idx , const size_t NPar, std::vector<double>& ParList, std::vector<double>& Res)
{
  //Create the cuSolver
  static cusolverDnHandle_t handle = NULL;
  static cublasHandle_t cublasHandle = NULL; // used in residual evaluation
//  cudaStream_t stream = NULL;
  if (handle==NULL)cusolverDnCreate(&handle);
  if (cublasHandle==NULL)cublasCreate(&cublasHandle);
//  cudaStreamCreate(&stream);

  assert(x.size() == y.size());
  if (ParList.size() != NPar) ParList.resize(NPar);

  Res.resize(x.size());
  VecScale(0.0,Res);
  auto N_Eq = f_idx - i_idx;
  //minimize |Ax - b| 
  const double * Vb = reinterpret_cast<const double *>(&y[i_idx]);
  const double * Vdata = reinterpret_cast<const double *>(&x[i_idx]);
  double * VSol = reinterpret_cast<double *>(&ParList[0]);
  double * VRes = reinterpret_cast<double *>(&Res[i_idx]);
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
  size_t size_A = NPar*N_Eq*sizeof(double);
  size_t size_M = NPar*NPar*sizeof(double);
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_M, size_M);
  cudaMalloc(&d_b, N_Eq*sizeof(double));
  cudaMalloc(&d_rh, NPar*sizeof(double));
  cudaMalloc(&d_data, N_Eq*sizeof(double));
  cudaMalloc(&d_res, N_Eq*sizeof(double));
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
  cudaMemcpy(d_b, Vb, N_Eq*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, d_b, N_Eq*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_data, Vdata, N_Eq*sizeof(double), cudaMemcpyHostToDevice);

  //Make Matrix A
  dim3 DimBlock (NPar,16);
  dim3 DimGrid (1, N_Eq/16+1);
  MakeMatrix<<<DimGrid, DimBlock>>>(d_data,d_A,NPar,N_Eq);

  const double alpha = 1.0;
  const double beta  = 0.0;

  cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NPar, NPar, N_Eq, &alpha, d_A, N_Eq, d_A, N_Eq, &beta, d_M, NPar);
/*
  //Make Matrix M = AT * A
  dim3 DimBlockB (NPar,NPar,16);
  dim3 DimGridB (1,1, N_Eq/16+1);
  MakeCoefficientMatrix1<<<DimGridB, DimBlockB>>>(d_A,d_B,NPar,N_Eq);

  dim3 DimBlockM (NPar, NPar);
  dim3 DimGridM (1,1);
  MakeCoefficientMatrix2<<<DimGridM, DimBlockM>>>(d_B,d_M,NPar,N_Eq);
*/
  //Make Vector Rh = AT * b
/*  dim3 DimBlockC (NPar,16);
  dim3 DimGridC (1, N_Eq/16+1);
  MakeRH1<<<DimGridC, DimBlockC>>>(d_b,d_B,d_C,NPar,N_Eq);

  dim3 DimBlockRH (NPar);
  dim3 DimGridRH (1);
  MakeRH2<<<DimGridRH, DimBlockRH>>>(d_C,d_rh,NPar,N_Eq);
*/

  cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NPar, 1 , N_Eq, &alpha, d_A, N_Eq, d_b, N_Eq, &beta, d_rh, NPar);
//  cusolverDnSetStream(handle, stream);
//  cublasSetStream(cublasHandle, stream);

  linearSolverCHOL(handle, NPar, d_M, NPar, d_rh, d_par);

  //Calculate residual

  const double beta2  = -1.0;
  cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N_Eq, 1 , NPar, &alpha, d_A, N_Eq, d_par, NPar, &beta2, d_res, N_Eq);
/*
  h_M = (double *)malloc(size_M);
  cudaMemcpy(h_M, d_M, size_M, cudaMemcpyDeviceToHost);
*/
  cudaMemcpy(VSol, d_par, NPar*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(VRes, d_res, N_Eq*sizeof(double), cudaMemcpyDeviceToHost);

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

} // ::dsp
