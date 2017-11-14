#ifndef FIDFACTORY_H
#define FIDFACTORY_H

/*===========================================================================*\

author: Matthias W. Smith , Ran Hong
email: mwmsith2@uw.edu , rhong@anl.gov
file:  fidfactory.h

notes:

  The class, FidFactory simulates FIDs using the Bloch equations and 
  numerical integration.

\*===========================================================================*/

//--- std includes ----------------------------------------------------------//
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <complex>

// //--- other includes --------------------------------------------------------//
#include <boost/numeric/odeint.hpp>
#include <armadillo>
#include "TFile.h"
#include "TTree.h"

// //--- project includes ------------------------------------------------------//
#include "dsp.h"

//--- namespaces ------------------------------------------------------------//

namespace fid {

namespace grad {

  std::string root_file = "~/.fid/sim_fids.root";
  std::string fid_branch = "fid";
  double min = -2000;
  double max = 2000;
  int poln_order = 2;
  std::vector<double> poln_coefs = {0.0, 0.0, 1.0, 0.0};
}


class FidFactory
{
 public:

  // ctors
  FidFactory();

  // dtors
  ~FidFactory();

  // member methods
  void IdealFid(std::vector<double>& wf, std::vector<double>& tm);
  void GradientFid_Bugao(std::vector<double>& wf, std::vector<double>& tm, std::vector<double>& fq);
  void SimulateFid(std::vector<double>& wf, std::vector<double>& tm);
  void GradientFid(const std::vector<double>& grad, std::vector<double>& wf);

  // Functions to set simulation parameters.
  void SetSeed(const int seed) { seed_ = seed; };
  void SetIntegrationStep(const double step) { integration_step_ = step; };

  void SetSignalToNoise(const double snr) { signal_to_noise_ = snr; };
  void SetAmplitude(const double amp) { amplitude_ = amp; };
  void SetBaseline(const double baseline) { baseline_ = baseline; };
  void SetStartTime(const double t0) { start_time_ = t0; };
  void SetDeltaTime(const double dt) { sample_time_ = dt; };
  void SetNumSamples(int n) { num_samples_ = n; };

  void SetLarmorFreq(const double f) { larmor_freq_ = f; };
  void SetMixdownFreq(const double f) { mixdown_freq_ = f; };
  void SetFidFreq(const double f) { larmor_freq_ = mixdown_freq_ + f; };
  void SetLowpassRatio(const double r) { lowpass_ratio_ = r; };
  void SetMixdownPhi(const double phi) { mixdown_phi_ = phi; };
  void SetInitialSpin(const std::vector<double>& s) { spin_0_ = s; };

  void SetGamma1(const double g) { gamma_1_ = g; };
  void SetGamma2(const double g) { gamma_2_ = g; };
  void SetGammaG(const double g) { gamma_g_ = g; };
  void SetRfOmega(const double o) { pulse_freq_ = o; };
  void SetRfDuration(const double t) { pulse_time_ = t; };

  void SetAddNoise(bool addnoise) { addnoise_ = addnoise; };
  void SetDiscrete(bool discrete) { discrete_ = discrete; };

  // accessors
  const int& seed() { return seed_; };
  const int& num_samples() { return num_samples_; };
  const double& start_time() { return start_time_; };
  const double& sample_time() { return sample_time_; };
  const double& integration_step() { return integration_step_; };

  const double& signal_to_noise() { return signal_to_noise_; };
  const double& amplitude() { return amplitude_; };
  const double& baseline() { return baseline_; };

  const double& mixdown_freq() { return mixdown_freq_; };
  const double& mixdown_phi() { return mixdown_phi_; };
  const double& lowpass_ratio() { return lowpass_ratio_; };
  const double& gamma_1() { return gamma_1_; };
  const double& gamma_2() { return gamma_2_; };
  const double& gamma_g() { return gamma_g_; };
  const double& pulse_freq() { return pulse_freq_; };
  const double& pulse_time() { return pulse_time_; };

  const std::vector<double>& spin_0() { return spin_0_; };

  const bool& discrete() { return discrete_; };
  const bool& addnoise() { return addnoise_; };

  // utility accessors
  double freq() { return larmor_freq_ - mixdown_freq_; };

  void PrintDiagnosticInfo();

 private:

  double ti_;
  double tf_;
  double dt_;
  int sim_to_fid_;
  int sim_length_;
  int printer_idx_;

  std::vector<double> s_;
  std::vector<double> spin_vec_;
  std::vector<double> time_vec_;
  std::vector<double> spin_sum_;
  std::vector<double> cos_cache_;
  std::vector<double> gradient_;

  // Simulation parameters (loaded from fid::sim namespace).
  const int padding_ = 100;
  int seed_ = 0;
  double integration_step_ = 2.0e-5;
  double signal_to_noise_ = 90000.0;   
  double amplitude_ = 2000.0;
  double baseline_ = 21000.0;

  int num_samples_ = 10000;
  double start_time_ = -1.0; 
  double sample_time_ = 0.001; 

  double mixdown_freq_ = 950.0;    
  double larmor_freq_ = 997.0; 
  double lowpass_ratio_ = 0.1;
  double mixdown_phi_ = 0.0; 
  std::vector<double> spin_0_ = {0.0, 0.0, 1.0};         

  double gamma_1_ = 0.05;  
  double gamma_2_ = 0.05;  
  double gamma_g_ = 1.0;  

  double pulse_freq_ = 125.0;  
  double pulse_time_ = 0.002;

  bool addnoise_ = true;
  bool discrete_ = false;

  // Low pass filter to extract mixed down signal.
  std::vector<double> LowPassFilter(std::vector<double>& s);

  // Function which returns time dependent Bfield.
  std::vector<double> Bfield(const double& t);

  // The time evolution equation for the fields.
  void Bloch(std::vector<double> const &s, std::vector<double> &dsdt, double t);

  // The integration monitor function
  void Printer(std::vector<double> const &s , double t);

  int num_sim_fids_;
  int zero_idx_;
  double d_grad_;

  TFile *pf_fid_;
  TTree *pt_fid_;
  std::vector<Double_t> wf_;

  int GetTreeIndex(double grad_strength);

  void LoadParams(std::string conf_file);
}; // FidFactory

} // ::fid

#endif
