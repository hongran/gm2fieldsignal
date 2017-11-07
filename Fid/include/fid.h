#ifndef FID_H
#define FID_H

/*============================================================================*\

author: Matthias W. Smith, Ran Hong
email: mwmsith2@uw.edu, rhong@anl.gov

notes:

  Defines a base class for FID classes.

\*===========================================================================*/

//--- std includes ----------------------------------------------------------//
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

//--- other includes --------------------------------------------------------//
//#include "boost/filesystem.hpp"
#include "TCanvas.h"
#include "TGraph.h"
#include "TF1.h"

//--- project includes ------------------------------------------------------//
//#include "params.h"
#include "dsp.h"

namespace fid {

// Enumerate the different methods of frequency extraction
enum Method { ZC=0, CN=1, AN=2, LZ=3, EX=4, PH=5, SN=6, PD=7,
              ZEROCOUNT,
              CENTROID,
              ANALYTICAL,
              LORENTZIAN,
              EXPONENTIAL,
              PHASE,
              SINUSOID,
	      PHASEDERIVITIVE
};

class Fid {

 public:

  // Default ctors/dtors.
  Fid();
  Fid(const std::vector<double>& wf, const std::vector<double>& tm);
  Fid(const std::vector<double>& wf);
  ~Fid() {};
  void Init(std::string Option = std::string("Standard"));
  
  // Simplified frequency extraction
  double GetFreq(const std::string& method_name);
  double GetFreq(const Method m = PH);
  double GetFreqError(const std::string& method_name);
  double GetFreqError(const Method m = PH);

  void CalcFreq();

  // accessors
  const std::vector<double>& wf() const { return wf_; };
  const std::vector<double>& tm() const { return tm_; };

  const std::vector<double>& res() const { return res_; };
  const std::vector<double>& psd() const { return psd_; };
  const std::vector<double>& phi() const { return phi_; };
  const std::vector<double>& env() const { return env_; };
  const std::vector<double>& fftfreq() const { return fftfreq_; };
  const std::vector<double>& filtered_wf () const { return filtered_wf_; }; // filtered waveform
  const std::vector<double>& baseline () const { return baseline_; }; // baseline
  const std::vector<dsp::cdouble>& fid_fft () const { return fid_fft_; }; //spectrum after fft
  const std::vector<double>& wf_im () const { return wf_im_; }; //harmonic conjugate of the waveform

  const double* freq_array() const {  return freq_array_;  }
  const double* freq_err_array() const { return freq_err_array_; };

  const double fid_time() const { return tm_[f_wf_] - tm_[i_wf_]; };
  //const double snr() const { return pow(max_amp_ / noise_, 2); };
  const double noise() const { return noise_; };
  const double snr() const { return snr_; };
  const double amp() const { return max_amp_; };
  const bool isgood() const { return health_ > 0.0; };
  const ushort health() const { return health_; };
  const int freq_method() const { return freq_method_; };

  const unsigned int& i_wf() { return i_wf_; };
  const unsigned int& f_wf() { return f_wf_; };
  const unsigned int& i_fft() { return i_fft_; };
  const unsigned int& f_fft() { return f_fft_; };

  const double& chi2() const { return chi2_; };
  const TF1&    f_fit() const { return f_fit_; };
  const TGraph& gr_time_series() const { return gr_time_series_; };
  const TGraph& gr_freq_series() const { return gr_freq_series_; };

  // Set parameters
  void SetParameter(const std::string& Name, double Value);
  void SetNoise(double NoiseValue);
  void SetBaseline(const std::vector<double>& bl);

  // Frequency Extraction Methods
  double CalcZeroCountFreq();
  double CalcCentroidFreq();
  double CalcAnalyticalFreq();
  double CalcLorentzianFreq();
  double CalcSoftLorentzianFreq();
  double CalcExponentialFreq();
  double CalcPhaseFreq();
  double CalcPhaseDerivFreq(int poln=3);
  double CalcSinusoidFreq();

  void CalcNoise();
  void CalcMaxAmp();      
  double CalcRms();      
  void CenterFid();
  void FindFidRange();
  void CallFFT();
  void CalcPowerEnvAndPhase();
  void BaselineCorrection();
  void ConstBaselineCorrection(double ConstBaseline);
  void CalcFftFreq();
  void GuessFitParams();
  void FreqFit(TF1& func);
 protected:
  
  // Private Member Variables
  unsigned int i_wf_; // start and stop of relevant data
  unsigned int f_wf_;
  unsigned int i_fft_;
  unsigned int f_fft_;
  unsigned int max_idx_fft_;

  // Waveform characteristics
  double mean_;
  double noise_;
  double max_amp_;
  double snr_;

//  double freq_;
//  double freq_err_;
  //Store freq from all methods
  double freq_array_[8];
  double freq_err_array_[8];
  double chi2_; // Store the most recent chi2
  ushort health_; // percentage between 0 and 100.

  //Parameters
  double edge_width_ = 2e-5;
  double edge_ignore_ = 6e-5;
  double start_amplitude_ = 0.37;
  double baseline_freq_thresh_ = 500.0;
  double filter_low_freq_ = 20000.0;
  double filter_high_freq_ = 80000.0;
  double fft_peak_width_ = 5000.0;
  double centroid_thresh_ = 0.01;
  double hyst_thresh_ = 0.7;
  double snr_thresh_ = 10.0;
  double len_thresh_ = 0.025;
  Method freq_method_ = PH;

  // For fits.
  std::vector<double> guess_;
  TF1 f_fit_;  
  TGraph gr_time_series_;
  TGraph gr_freq_series_;

  //FFT Done Flag
  bool FFTDone = false;

  // bigger data arrays
  std::vector<double> wf_; //waveform
  std::vector<double> tm_; //time stamps
  std::vector<double> filtered_wf_; // filtered waveform
  std::vector<double> baseline_; // baseline
  std::vector<dsp::cdouble> fid_fft_; //spectrum after fft
  std::vector<double> wf_im_; //harmonic conjugate of the waveform
  std::vector<double> psd_; //power density
  std::vector<double> env_; //envelope
  std::vector<double> phi_; //monotonic phase
  std::vector<double> fftfreq_; // fft frequency stamp
  std::vector<double> res_; //residual of fit
//  std::vector<double> temp_; // for random transformations

  // Private Member Functions  
  // init function to be called after wf_ and tm_ are set.

  // internal utility functions
  Method ParseMethod(const std::string& m);
  
}; // Fid
 
} // fid

#endif
