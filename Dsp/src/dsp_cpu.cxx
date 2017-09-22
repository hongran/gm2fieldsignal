#include "dsp.h"
#include "fftw3.h"
#include "TF1.h"
#include "TGraph.h"
#include <cassert>
#include <chrono>

namespace dsp {

std::vector<double>& VecAdd(std::vector<double>& a, std::vector<double>& b)
{
  assert(a.size() == b.size());

  transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>());
  return a;
}

std::vector<double>& VecSubtract(std::vector<double>& a, std::vector<double>& b)
{
  assert(a.size() == b.size());

  transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<double>());
  return a;
}

// A template function to handle vector multiplying with a scalar.
std::vector<double>& VecScale(double c, std::vector<double>& a)
{
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c * (*it);
  }

  return a;
}

std::vector<double>& VecShift(double c, std::vector<double>& a)
{
  for (auto it = a.begin(); it != a.end(); ++it){
    *it = c + (*it);
  }

  return a;
}

double VecChi2(std::vector<double>& a){
  double chi2 = 0;
  for (auto it = a.begin(); it != a.end(); ++it){
    chi2 += (*it)*(*it);
  }
  return chi2;
}

//FFT

std::vector<cdouble> ifft(const std::vector<cdouble>& v)
{
  // Grab some useful constants.
  int N = v.size();
  double Nroot = std::sqrt(N);

  // Instantiate the result vector.
  std::vector<cdouble> wfm_vec(N, cdouble(0.0, 0.0));
  auto fft_vec = v;

  fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[0]);  
  fftw_complex *wfm_ptr = reinterpret_cast<fftw_complex *>(&wfm_vec[0]);  

  // Plan and execute the inverse fft (+1 == exponent).
  auto plan = fftw_plan_dft_1d(N, fft_ptr, wfm_ptr, +1, FFTW_ESTIMATE);  
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // fftw is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
    *it /= Nroot;
  }

  return wfm_vec;
}

std::vector<cdouble> rfft(const std::vector<double> &v)
{
  // Grab some useful constants.
  int N = v.size();  
  int n = N / 2 + 1;  // size of rfft
  double Nroot = std::sqrt(N);

  //Test Time cost
  //auto t0 = std::chrono::high_resolution_clock::now();

  // Instantiate the result vector.
  std::vector<cdouble> fft_vec(n, 0.0);
  auto wfm_vec = v; // copy waveform since fftw destroys it

  fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[0]);

  // Plan and execute the fft.
  auto plan = fftw_plan_dft_r2c_1d(N, &wfm_vec[0], fft_ptr, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  for (auto it = fft_vec.begin(); it != fft_vec.end(); ++it) {
    *it /= Nroot;
  }
  /*auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
  double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
  std::cout << "Time = "<<t<<std::endl;
  */

  return fft_vec;
}


std::vector<double> irfft(const std::vector<cdouble>& fft, bool is_odd)
{
  // Grab some useful constants.
  int n = fft.size();
  int N = 2 * (n - 1) + 1 * is_odd;
  double Nroot = std::sqrt(N);

  // Instantiate the result vector.
  std::vector<double> wfm_vec(N, 0.0);
  std::vector<cdouble> fft_vec = fft;

  fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[0]);

  // Plan and execute the fft.
  auto plan = fftw_plan_dft_c2r_1d(N, fft_ptr, &wfm_vec[0], FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // fftw is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
  	*it /= Nroot;
  }

  return wfm_vec;
}

std::vector<double> norm(const std::vector<double>& v)
{
  // Allocate the memory
  std::vector<double> res;
  res.reserve(v.size());

  // Iterate and push back the norm.
  for (auto it = v.begin(); it < v.end(); ++it) {
    res.push_back(std::norm(*it));
  }

  return res;
}

std::vector<double> norm(const std::vector<cdouble>& v)
{
  // Allocate the memory
  std::vector<double> res;
  res.reserve(v.size());

  // Iterate and push back the norm.
  for (auto it = v.begin(); it < v.end(); ++it) {
    res.push_back(std::norm(*it));
  }

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
  std::vector<double> phase(wf_re.size(), 0.0);
  
  // Calculate the modulo-ed phase
  std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), phase.begin(),
                 [](double re, double im) { return std::atan2(im, re); });
  
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
 	// Set the envelope function
	std::vector<double> env(wf_re.size(), 0.0);

	std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), env.begin(),
    			   [](double r, double i) { return std::sqrt(r*r + i*i); });

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

int linear_fit(const std::vector<double>& x, const std::vector<double>& y, const unsigned int i_idx, const unsigned int f_idx ,  const size_t NPar, std::vector<double>& ParList, std::vector<double>& Res){
  TGraph gr_time_series_ = TGraph(f_idx - i_idx, &x[i_idx], &y[i_idx]);
  Res.resize(x.size());
  // Now set up the polynomial phase fit
  char fcn[20];
  sprintf(fcn, "pol%d", static_cast<int>(NPar)-1);
  TF1 f_fit_ = TF1("f_fit", fcn,x[i_idx],x[f_idx]);

  gr_time_series_.Fit(&f_fit_,"QREX0");

  for (unsigned int i=0;i<NPar;i++){
    ParList[i] = f_fit_.GetParameter(i);
  }

  for (unsigned int i=0;i<Res.size();i++){
    Res[i] = y[i] - f_fit_.Eval(x[i]);
  }

  return 0;
}
 
} // ::dsp
