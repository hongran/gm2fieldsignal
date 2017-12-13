#ifndef DSP_H
#define DSP_H

/*===========================================================================*\

author: Matthias W. Smith and Ran Hong
email: mwmsith2@uw.edu, rhong@anl.gov

notes: 

	This header defines some commonly used mathematical and digital 
	signal processing functions for the FID library.

\*===========================================================================*/

//--- std includes ----------------------------------------------------------//
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <complex>
#include <ctgmath>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

//--- other includes --------------------------------------------------------//
#include <armadillo>

//--- project includes ------------------------------------------------------//


namespace dsp
{
// Aliases
typedef std::complex<double> cdouble;

// constants
const double kTau = 2 * M_PI;
const double kMaxPhaseJump = 4.71;

//--- linalg template functions ---------------------------------------------//

//floor function
template <typename T> void floor(std::vector<T>& v) 
{
  for (auto it = v.begin(); it != v.end(); ++it) {
    *it = std::floor(*it);
  }
}

// Construct a range from params first, step, last
template<typename T> std::vector<T> construct_range(const T& i, const T& f, const T& d) {
	std::vector<T> res;
	for (T x = i; x <= f; x += d){
	  res.push_back(x);
	}
	return res;
}

// Construct a range from vector <first, last, step>
template<typename T> std::vector<T> construct_range(const std::vector<T> &range_vec) {
	std::vector<T> res;
	for (T x = range_vec[0]; x <= range_vec[1]; x += range_vec[2]){
	  res.push_back(x);
	}
	return res;
}

template<typename T> std::vector<T> construct_linspace(const T& i, const T& f, const int& n) {
    std::vector<T> res;
    int N;
    if (n == 0) {
      N = f - i;
    } else {
      N = n;
    }
    double d = (f - i) / (n - 1);
    for (int j = 0; j < n; ++j) {
        res.push_back(i + j*d);
    }
    return res;
}

template<typename T> std::vector<T> construct_linspace(const std::vector<T>& vals) {
    std::vector<T> res;
    int n = (int)(vals[2] + 0.5);
    double d = (vals[1] - vals[0]) / (n - 1);
    for (int j = 0; j < n; ++j) {
        res.push_back(vals[0] + j*d);
    }
    return res;
}

template <typename T> void cross(const std::vector<T>& u, 
                  const std::vector<T>& v, 
                  std::vector<T>& res)
{
    res[0] = u[1] * v[2] - u[2] * v[1];
    res[1] = u[2] * v[0] - u[0] * v[2];
    res[2] = u[0] * v[1] - u[1] * v[0];
}

//Mean calculation based on whole vector
template <typename T> double mean(const T& v) {
  
  double sum = 0.0;
  for(int i = 0; i< v.size(); i++) sum += v[i];

  return sum/(double)v.size();
}

//Mean calculation based on whole array
template <typename T> double mean_arr(const T& v) {

  double size = sizeof(v)/sizeof(v[0]);
  double sum = 0.0;
  for(int i = 0; i< size; i++) sum += v[i];

  return sum/size;
}

// Standard deviation calculation based on whole vector.
template <typename T> double stdev(const T& v) {
    auto x1 = std::accumulate(v.begin(), v.end(), 0.0, 
        [](double x, double y) { return x + y; });

	auto x2 = std::accumulate(v.begin(), v.end(), 0.0, 
		[](double x, double y) { return x + y*y; });

  double N = (double) std::distance(v.begin(), v.end());
	return std::sqrt(x2/N - (x1/N) * (x1/N));
}

//Standard deviation calculation based on whole array
template <typename T> double stdev_arr(const T& a){
  
  double mean = mean_arr(a);
  double size = sizeof(a)/sizeof(a[0]);
  double sum = 0.0;

  for(int i = 0;i < size; i++) sum += pow(a[i]-mean,2);

  double var = sum/size;
  double sd = sqrt(var);  

  return sd;
 } 

// Standard deviation calculation based on start/stop iterators.
template <typename T> double stdev(const T& begin, const T& end) {
    auto x1 = std::accumulate(begin, end, 0.0, 
        [](double x, double y) { return x + y; });

	auto x2 = std::accumulate(begin, end, 0.0, 
		[](double x, double y) { return x + y*y; });

  double N = (double) std::distance(begin, end);
  return std::sqrt(x2/N - (x1/N) * (x1/N));
}

// Add white noise to an array.
template <typename T> void addwhitenoise(std::vector<T>& v, T snr) {
  static std::default_random_engine gen(clock());
  std::normal_distribution<T> nrm;

  T mean = 0.0;
  T min = 0.0;
  T max = 0.0;
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

  T amp = max > min ? max : min;
  T scale = amp / sqrt(snr);

  for (auto &x : v){
    x += nrm(gen) * scale;
  }
}

//low pass filter
template <typename T>
std::vector<T> lowpass(const std::vector<T>& v, double cut_idx=-1, int n=3) {
	// A simple Butterworth n-order filter.
	if (cut_idx == -1) cut_idx = v.size() / 2;
	std::vector<T> filtered_wf = v;

	std::transform(v.begin(), v.end(), filtered_wf.begin(), // lambda filter
				  [cut_idx, n](double x) { 
				  	return sqrt(1.0 / (1.0 + pow(x / cut_idx, 2*n))); 
				  });

	return filtered_wf;
}

template <typename T>
arma::Col<T> rconvolve(const arma::Col<T>& v, int idx=0) {
	int ridx = v.n_elem - idx;
	arma::Col<T> rv(arma::flipud(v));
	arma::Col<T> res(v.n_elem, arma::fill::zeros);

	if (idx > ridx) {
		std::transform(v.begin() + idx, v.end(), rv.begin(), res.begin(),
			[](T z1, T z2) { return z1 * std::conj(z2); });
	} else {
		std::transform(rv.begin() + ridx, rv.end(), v.begin(), res.begin(),
			[](T z2, T z1) { return z1 * std::conj(z2); });
	}
	return res;
} 

//Function declare 
std::vector<double>& VecAdd(std::vector<double>& a, std::vector<double>& b);
std::vector<double>& VecSubtract(std::vector<double>& a, std::vector<double>& b);
std::vector<double>& VecScale(double c, std::vector<double>& a);
std::vector<double>& VecShift(double c, std::vector<double>& a);
double VecChi2(std::vector<double>& a);

std::vector<double> normalized_gradient(int npoints, int poln=1); 

std::vector<cdouble> ifft(const std::vector<cdouble>& v);
std::vector<cdouble> rfft(const std::vector<double>& v);
std::vector<double> irfft(const std::vector<cdouble>& v, bool is_odd);

std::vector<double> hilbert(const std::vector<double>& v);
std::vector<double> psd(const std::vector<double>& v);

std::vector<double> norm(const std::vector<double>& v);
std::vector<double> norm(const std::vector<cdouble>& v);

std::vector<double> fftfreq(const std::vector<double>& tm);
std::vector<double> fftfreq(const int N, const double dt);

std::vector<cdouble> window_filter(const std::vector<cdouble>& spectrum, const std::vector<double>& time, double low, double high);

std::vector<double> phase(const std::vector<double>& v);
std::vector<double> phase(const std::vector<double>& wf_re, 
                          const std::vector<double>& wf_im);

std::vector<double> envelope(const std::vector<double>& v);
std::vector<double> envelope(const std::vector<double>& wf_re, 
                             const std::vector<double>& wf_im);	
/*
arma::cx_mat wvd_cx(const std::vector<double>& v, bool upsample=false);
arma::mat wvd(const std::vector<double>& v, bool upsample=false);
*/

std::vector<double> savgol3(const std::vector<double>& v);
std::vector<double> savgol5(const std::vector<double>& v);
std::vector<double> convolve(const std::vector<double>& v, 
                             const std::vector<double>& filter);
int convolve(const std::vector<double>& v, 
             const std::vector<double>& filter, 
             std::vector<double>& res);

int linear_fit(const std::vector<double>& x, const std::vector<double>& y, const unsigned int i_idx, const unsigned int f_idx ,  const size_t NPar, std::vector<double>& ParList, std::vector<double>& Res);

class IntegratedProcessor{
  public:
    IntegratedProcessor(unsigned int BatchNum, unsigned int len);
    int SetFilters(double low, double high, double baseline_thresh );
    int SetEdges(double ignore, double width);
    void AnaSwith(bool sw);
    int Process(const std::vector<double>& wf,const std::vector<double>& tm, std::vector<double>& freq,
	std::vector<double>& filtered_wf, std::vector<double>& wf_im, std::vector<double>& baseline,
	std::vector<double>& psd, std::vector<double>& phi , std::vector<double>& env,
	std::vector<unsigned int>& iwf, std::vector<unsigned int>& fwf,
	std::vector<unsigned int> max_idx_fft,std::vector<unsigned int>& i_fft,std::vector<unsigned int>& f_fft, 
	std::vector<double> max_amp,std::vector<unsigned short>& health);
  protected:
    unsigned int Length;
    unsigned int NBatch;
    double WindowFilterLow = 20000.0;
    double WindowFilterHigh = 80000.0;
    double Baseline_Freq_Thresh = 500.0;
    double edge_ignore = 6e-5;
    double edge_width = 2e-5;
    double start_amplitude = 0.37;
    double fft_peak_width = 5000.0;
    bool FreqAnaSwitch = true;
    double Freq;
    double FreqErr;
    //
    //not sure about this

    //FFT plans

    cufftHandle planD2Z;
    cufftHandle planZ2D;
    cufftHandle planZ2Z;
};

} // ::dsp


#endif
