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
  Fid(const std::vector<double>& wf, double dt, unsigned int tNBatch, unsigned int tSize);
  Fid(const std::vector<double>& wf);
  ~Fid() {};
  void Init(std::string Option = std::string("Standard"));

  //Waveform set function
  SetWf(const std::vector<double>& wf, double dt, unsigned int tNBatch, unsigned int tSize);
  
  // Simplified frequency extraction
  double GetFreq(const std::string& method_name, unsigned int Index = 0);
  double GetFreq(const Method m = PH, unsigned int Index = 0);
  double GetFreqError(const std::string& method_name, unsigned int Index = 0);
  double GetFreqError(const Method m = PH, unsigned int Index = 0);

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

  std::vector<std::vector<double>> freq_array() const {  return freq_array_;  }
  std::vector<std::vector<double>> freq_err_array() const { return freq_err_array_; };

  //const double snr() const { return pow(max_amp_ / noise_, 2); };
  const double noise(unsigned int Index=0) const { return noise_[Index]; };
  const double snr(unsigned int Index=0) const { return snr_[Index]; };
  const double amp(unsigned int Index=0) const { return max_amp_[Index]; };
  const bool isgood(unsigned int Index=0) const { return health_[Index] > 0.0; };
  const ushort health(unsigned int Index=0) const { return health_[Index]; };
  const int freq_method(unsigned int Index=0) const { return freq_method_[Index]; };
  const double fid_time(unsigned int Index=0) const { return tm_[Index*fid_size+f_wf_[Index]] - tm_[Index*fid_size+i_wf_[Index]]; };
  const double act_length_(unsigned int Index=0) const { return act_length_[Index]; };

  const unsigned int& i_wf(unsigned int Index=0) { return i_wf_[Index]; };
  const unsigned int& f_wf(unsigned int Index=0) { return f_wf_[Index]; };
  const unsigned int& i_fft(unsigned int Index=0) { return i_fft_[Index]; };
  const unsigned int& f_fft(unsigned int Index=0) { return f_fft_[Index]; };

  const double& chi2(unsigned int Index=0) const { return chi2_[Index]; };
  const TF1&    f_fit(unsigned int Index=0) const { return f_fit_[Index]; };
  const TGraph& gr_time_series(unsigned int Index=0) const { return gr_time_series_[Index]; };
  const TGraph& gr_freq_series(unsigned int Index=0) const { return gr_freq_series_[Index]; };

  // Set parameters
  void SetParameter(const std::string& Name, double Value);
  //Set noise and baselien by user
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
  
  unsigned int NBatch;	//number of batches, or number of fids
  unsigned int fid_size; //size (length) of each fid (batch)

  // Private Member Variables
  std::vector<unsigned> int i_wf_; // start and stop of relevant data
  std::vector<unsigned> int f_wf_;
  std::vector<unsigned> int i_fft_;
  std::vector<unsigned> int f_fft_;
  std::vector<unsigned int> max_idx_fft_;

  // Waveform characteristics
  std::vector<double> mean_;
  std::vector<double> noise_;
  std::vector<double> snr_;
  std::vector<double> max_amp_;
  std::vector<double> act_length_; //Fid active length, when max amp decays to its 1/e
  std::vector<ushort> health_; // percentage between 0 and 100.

//  double freq_;
//  double freq_err_;
  //Store freq from all methods
  std::vector<std::vector<double>> freq_array_;
  std::vector<std::vector<double>> freq_err_array_;
  std::vector<double> chi2_; // Store the most recent chi2

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
  std::vector<TF1> f_fit_;  
  std::vector<TGraph> gr_time_series_;
  std::vector<TGraph> gr_freq_series_;

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
