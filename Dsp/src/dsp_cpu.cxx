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

//FFT

std::vector<cdouble> Processor::ifft(const std::vector<cdouble>& v,unsigned int N, unsigned int NBatch)
{
  // Grab some useful constants.
//  int N = v.size();
  double Nroot = std::sqrt(N);

  // Instantiate the result vector.
  std::vector<cdouble> wfm_vec(N*NBatch, cdouble(0.0, 0.0));
  auto fft_vec = v;

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[j*N]);  
    fftw_complex *wfm_ptr = reinterpret_cast<fftw_complex *>(&wfm_vec[j*N]);  

    // Plan and execute the inverse fft (+1 == exponent).
    auto plan = fftw_plan_dft_1d(N, fft_ptr, wfm_ptr, +1, FFTW_ESTIMATE);  
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }
  // fftw is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
    *it /= Nroot;
  }

  return wfm_vec;
}

std::vector<cdouble> Processor::rfft(const std::vector<double> &v,unsigned int N, unsigned int NBatch)
{
  // Grab some useful constants.
//  int N = v.size();  
  unsigned int n = N / 2 + 1;  // size of rfft
  double Nroot = std::sqrt(N);

  //Test Time cost
  //auto t0 = std::chrono::high_resolution_clock::now();

  // Instantiate the result vector.
  std::vector<cdouble> fft_vec(n*NBatch, 0.0);
  auto wfm_vec = v; // copy waveform since fftw destroys it

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[j*N]);

    // Plan and execute the fft.
    auto plan = fftw_plan_dft_r2c_1d(N, &wfm_vec[j*N], fft_ptr, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

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


std::vector<double> Processor::irfft(const std::vector<cdouble>& fft, unsigned int N, unsigned int NBatch)
{
  // Grab some useful constants.
  unsigned int n = N/2+1;
//  int N = 2 * (n - 1) + 1 * is_odd;
  double Nroot = std::sqrt(N);

  // Instantiate the result vector.
  std::vector<double> wfm_vec(N*NBatch, 0.0);
  std::vector<cdouble> fft_vec = fft;

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec[j*NBatch]);

    // Plan and execute the fft.
    auto plan = fftw_plan_dft_c2r_1d(n, fft_ptr, &wfm_vec[j*NBatch], FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  // fftw is unnormalized, so we need to fix that.
  for (auto it = wfm_vec.begin(); it != wfm_vec.end(); ++it) {
  	*it /= Nroot;
  }

  return wfm_vec;
}

std::vector<double> Processor::hilbert(const std::vector<double>& v ,unsigned int N, unsigned int NBatch)
{
  // Return the call to the fft version.
  auto fft_vec = rfft(v , N, NBatch);

  // Zero out the constant term.
  fft_vec[0] = cdouble(0.0, 0.0);

  // Multiply in the -i.
  for (auto it = fft_vec.begin(); it != fft_vec.end(); ++it) {
    *it = cdouble((*it).imag(), -(*it).real());
  }


  // Reverse the fft.
  return irfft(fft_vec, N,NBatch);
}

std::vector<double> Processor::psd(const std::vector<cdouble>& fft_vec)
{
  // Get the norm of the fft as that is the power.
  return norm(fft_vec);
}

//Window Filter

std::vector<cdouble> Processor::window_filter(const std::vector<cdouble>& spectrum, const std::vector<double>& freq, double low, double high,unsigned int N, unsigned int NBatch)
{
  auto output = spectrum;
  auto interval = freq[1] - freq[0];
  unsigned int i_start = std::floor((low - freq[0])/interval);
  unsigned int i_end = std::floor((high - freq[0])/interval);
  for (unsigned int j=0;j<NBatch;j++){
    for (unsigned int i=j*N ; i<(j+1)*N ; i++){
      if (i<i_start || i>i_end){
	output[i] = cdouble(0.0, 0.0);
      }
    }
  }
  return output;
}

// Calculates the phase by assuming the real signal is harmonic.
std::vector<double> Processor::phase(const std::vector<double>& wf_re, 
                               const std::vector<double>& wf_im,unsigned int N, unsigned int NBatch)
{
  std::vector<double> phase(wf_re.size(), 0.0);
  
  // Calculate the modulo-ed phase
  std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), phase.begin(),
                 [](double re, double im) { return std::atan2(im, re); });

  for (unsigned int i=0;i<NBatch;i++){
    int k=0;
    double previous = phase[i*N];
    for (unsigned int j = i*N + 1; j < (i+1)*N; j++) {
      double u = phase[j] - previous;
      previous = phase[j];
      if (-u > kMaxPhaseJump){
	k++;
      }
      if (u > kMaxPhaseJump){
	k--;
      }
      phase[j]+=k*kTau;
    }
  }

  /*
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
  }*/
  
  return phase;
}

std::vector<double> Processor::envelope(const std::vector<double>& wf_re, const std::vector<double>& wf_im)
{
  // Set the envelope function
  std::vector<double> env(wf_re.size(), 0.0);

  std::transform(wf_re.begin(), wf_re.end(), wf_im.begin(), env.begin(),
      [](double r, double i) { return std::sqrt(r*r + i*i); });

  return env;
}

void Processor::FindFitRange(unsigned int Length, unsigned int NBatch,double start_amplitude, double fft_peak_width,
    std::vector<double>& psd,std::vector<double>&env,double interval,
    std::vector<unsigned int>& iwf, std::vector<unsigned int>& fwf,
    std::vector<unsigned int>& max_idx_fft,std::vector<unsigned int>& i_fft,std::vector<unsigned int>& f_fft, 
    std::vector<double>& max_amp,std::vector<unsigned short>& health)
{
  max_amp.resize(NBatch);
  iwf.resize(NBatch);
  fwf.resize(NBatch);
  for (unsigned int i = 0; i<NBatch ; i++){
    auto it = std::max_element(env.begin()+i*Length,env.begin()+(i+1)*Length);
    double max = *it;
    max_amp[i] = max;
    unsigned int k = std::distance(env.begin()+i*Length,it);
    iwf[i] = k;
    fwf[i] = k;
    while (k<Length){
      if (env[i*Length+k]<max*start_amplitude){
	fwf[i] = k;
	break;
      }
      k++;
    }
  }

  for(unsigned int i=0; i<NBatch; i++)
  {
    if(iwf[i]>Length*0.95||iwf[i]>=fwf[i])
    {
    //std::cout<<i<<" "<<iwf[i]<<" "<<fwf[i]<<" "<<max_amp[i]<<std::endl;
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

  for (unsigned int i = 0; i<NBatch ; i++){
    auto it = std::max_element(psd.begin()+i*m,psd.begin()+(i+1)*m);
    unsigned int k = std::distance(psd.begin()+i*m,it);
    max_idx_fft[i] = k;
    if(k<=fft_peak_index_width) i_fft[i]=1;
    else i_fft[i]=k-fft_peak_index_width;
    f_fft[i]=k+fft_peak_index_width;
    if(f_fft[i]>m) f_fft[i]=m;
  }

}

// Helper function to get frequencies for FFT
std::vector<double> fftfreq(const std::vector<double>& tm,unsigned int N, unsigned int NBatch) 
{
//  int N = tm.size();
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
    freq.resize(m*NBatch);
    for(unsigned int j=0; j<NBatch; j++){ 
      for (unsigned int i = 0; i < m; i++) {
	freq[i+j*m] = i / (dt * N);
      }
    }
  } else {
    m = (N+1)/2;
    freq.resize(m*NBatch);
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
  Res.resize(x.size());
  for (unsigned int j=0;j<NBatch;j++){
    TGraph gr_time_series_ = TGraph(f_idx[j] - i_idx[j], &x[j*Length+i_idx[j]], &y[j*Length+i_idx[j]]);
    // Now set up the polynomial phase fit
    char fcn[20];
    sprintf(fcn, "pol%d", static_cast<int>(NPar)-1);
    TF1 fit_ = TF1("f_fit", fcn,x[j*Length+i_idx[j]],x[j*Length+f_idx[j]]);

    gr_time_series_.Fit(&fit_,"QREX0");

    for (unsigned int i=0;i<NPar;i++){
      ParLists[j][i] = fit_.GetParameter(i);
    }

    for (unsigned int i=0;i<Length;i++){
      Res[j*Length+i] = y[j*Length+i] - fit_.Eval(x[j*Length+i]);
    }
  }
  return 0;
}
 
//class IntegratedProcessor
/*
IntegratedProcessor::IntegratedProcessor(unsigned int len, unsigned int BatchNum):
  Length(len),NBatch(BatchNum)
{}


int IntegratedProcessor::SetFilterWindow(double low, double high )
{
  return 0;
}

int IntegratedProcessor::Process(const std::vector<double>& wf,const std::vector<double>& tm, std::vector<double>& freq,
    std::vector<double>& fwf, std::vector<double>& iwf, std::vector<double>& baseline,
    std::vector<double>& psd, std::vector<double>& phi , std::vector<double>& env)
{
  return 0;
}*/

//IntegratedProcessor Functions
IntegratedProcessor::IntegratedProcessor(unsigned int BatchNum, unsigned int len):
  Length(len),NBatch(BatchNum)
{}

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
  //constant
  double Nroot=std::sqrt(Length);
  //rfft****************************************************
  std::vector<cdouble> fid_fft(m*NBatch, 0.0);
  auto wfm_vec = wf; // copy waveform since fftw destroys it

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fid_fft[j*Length]);

    // Plan and execute the fft.
    auto plan = fftw_plan_dft_r2c_1d(Length, &wfm_vec[j*Length], fft_ptr, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  for (auto it = fid_fft.begin(); it != fid_fft.end(); ++it) {
    *it /= Nroot;
  }
  //window_filter*******************************************
  std::vector<cdouble> fid_fft_filtered = fid_fft;
  auto interval = freq[1] - freq[0];
  unsigned int i_start = std::floor((WindowFilterLow - freq[0])/interval);
  unsigned int i_end = std::floor((WindowFilterHigh - freq[0])/interval);
  for (unsigned int j=0;j<NBatch;j++){
    for (unsigned int i=j*Length ; i<(j+1)*Length ; i++){
      if (i<i_start || i>i_end){
	fid_fft_filtered[i] = cdouble(0.0, 0.0);
      }
    }
  }
  //irfft***************************************************
  filtered_wf.resize(Length*NBatch);
  auto fft_vec1 = fid_fft_filtered;

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec1[j*Length]);  
    fftw_complex *wfm_ptr = reinterpret_cast<fftw_complex *>(&filtered_wf[j*Length]);  

    // Plan and execute the inverse fft (+1 == exponent).
    auto plan = fftw_plan_dft_1d(m, fft_ptr, wfm_ptr, +1, FFTW_ESTIMATE);  
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }
  // fftw is unnormalized, so we need to fix that.
  for (auto it = filtered_wf.begin(); it != filtered_wf.end(); ++it) {
    *it /= Nroot;
  }
  //Imaginary waveform
  wf_im.resize(Length*NBatch); 
  auto fft_vec2 = fid_fft_filtered;

  // Multiply in the -i.
  for (auto it = fft_vec2.begin(); it != fft_vec2.end(); ++it) {
    *it = cdouble((*it).imag(), -(*it).real());
  }

  for (unsigned int j=0;j<NBatch;j++){
    fftw_complex *fft_ptr = reinterpret_cast<fftw_complex *>(&fft_vec2[j*Length]);  
    fftw_complex *wfm_ptr = reinterpret_cast<fftw_complex *>(&wf_im[j*Length]);  

    // Plan and execute the inverse fft (+1 == exponent).
    auto plan = fftw_plan_dft_1d(m, fft_ptr, wfm_ptr, +1, FFTW_ESTIMATE);  
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }
  // fftw is unnormalized, so we need to fix that.
  for (auto it = wf_im.begin(); it != wf_im.end(); ++it) {
    *it /= Nroot;
  }
  psd = norm(fid_fft);
  env.resize(Length*NBatch);
  std::transform(filtered_wf.begin(), filtered_wf.end(), wf_im.begin(), env.begin(),
      [](double r, double i) { return std::sqrt(r*r + i*i); });
  phi.resize(Length*NBatch);
  std::transform(filtered_wf.begin(), filtered_wf.end(), wf_im.begin(), phi.begin(),
                 [](double re, double im) { return std::atan2(im, re); });

  for (unsigned int i=0;i<NBatch;i++){
    int k=0;
    double previous = phi[i*Length];
    for (unsigned int j = i*Length + 1; j < (i+1)*Length; j++) {
      double u = phi[j] - previous;
      previous = phi[j];
      if (-u > kMaxPhaseJump){
	k++;
      }
      if (u > kMaxPhaseJump){
	k--;
      }
      phi[j]+=k*kTau;
    }
  }
  //Find Fit ranges
  max_amp.resize(NBatch);
  iwf.resize(NBatch);
  fwf.resize(NBatch);
  for (unsigned int i = 0; i<NBatch ; i++){
    auto it = std::max_element(env.begin()+i*Length,env.begin()+(i+1)*Length);
    double max = *it;
    max_amp[i] = max;
    unsigned int k = std::distance(env.begin()+i*Length,it);
    iwf[i] = k;
    fwf[i] = k;
    while (k<Length){
      if (env[i*Length+k]<max*start_amplitude){
	fwf[i] = k;
	break;
      }
      k++;
    }
  }

  for(unsigned int i=0; i<NBatch; i++)
  {
    if(iwf[i]>Length*0.95||iwf[i]>=fwf[i])
    {
    //std::cout<<i<<" "<<iwf[i]<<" "<<fwf[i]<<" "<<max_amp[i]<<std::endl;
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
//  auto interval = freq[1] - freq[0];
  unsigned int fft_peak_index_width = static_cast<int>(fft_peak_width/interval);

  for (unsigned int i = 0; i<NBatch ; i++){
    auto it = std::max_element(psd.begin()+i*m,psd.begin()+(i+1)*m);
    unsigned int k = std::distance(psd.begin()+i*m,it);
    max_idx_fft[i] = k;
    if(k<=fft_peak_index_width) i_fft[i]=1;
    else i_fft[i]=k-fft_peak_index_width;
    f_fft[i]=k+fft_peak_index_width;
    if(f_fft[i]>m) f_fft[i]=m;
  }

  //Fit
  linear_fit(tm,phi, iwf, fwf , NFitPar,NBatch,Length, FitPars, ResidualOut);
  //Fill Freq Res Array
  for (unsigned int i=0;i<NBatch;i++){
    FreqResArray[i][7] = FitPars[i][1] / kTau;
  }
  return 0;
}
//common
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
