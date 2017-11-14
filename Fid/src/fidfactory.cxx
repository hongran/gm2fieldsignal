#include "fidfactory.h"
#include "dsp.h"
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace fid {

FidFactory::FidFactory()
{
  // Set the start/stop times.
  ti_ = start_time_;
  tf_ = ti_ + sample_time_ * (num_samples_ + padding_);

  // Calculate the decimation ratio.
  sim_to_fid_ = (tf_ - ti_) / (integration_step_ * (num_samples_ + padding_)) + 0.5; 

  // If zero, more less integration steps than sampling times were requested.
  if (sim_to_fid_ == 0) {

    std::cout << "WARNING: The given integration step was larger than the ";
    std::cout << "sampling time, so the sampling time, " << sample_time_;
    std::cout << ", will be used instead." << std::endl;

    sim_to_fid_ = 1;
    dt_ = sample_time_;

  } else {

    dt_ = sample_time_ / sim_to_fid_;
  
    if (dt_ != integration_step_) {
      std::cout << "WARNING: The given integration time step was not an even";
      std::cout << " divisor of the sampling rate, so it has been rounded to ";
      std::cout << dt_ << std::endl;
    }
  }

  // Set number of simulation samples.
  sim_length_ = sim_to_fid_ * (num_samples_ + padding_);
  printer_idx_ = 0;

  // open the default root file
  pf_fid_ = new TFile(grad::root_file.c_str());

  if (pf_fid_->IsOpen()) {
    pt_fid_ = (TTree *)pf_fid_->Get("t");
  
    wf_.resize(num_samples_);
    pt_fid_->SetBranchAddress(grad::fid_branch.c_str(), &wf_[0]);
    pt_fid_->GetEntry(0);

    num_sim_fids_ = pt_fid_->GetEntries();
    d_grad_ = (grad::max - grad::min) / num_sim_fids_;
    zero_idx_ = -1 * grad::min / d_grad_ + 0.5;
  }
}


FidFactory::~FidFactory()
{
  // closing the TFile will take care of the TTree pointer
  pf_fid_->Close();
  delete pf_fid_;
}


void FidFactory::LoadParams(std::string conf_file)
{
  // using directives
  using boost::property_tree::ptree;

  // Declare and load JSON params.
  ptree pt;
  read_json(conf_file, pt);
  pt = pt.get_child("fid");

  // general fid parameters
  num_samples_ = pt.get<int>("sim.num_samples", num_samples_);
  start_time_ = pt.get<double>("sim.start_time", start_time_);
  sample_time_ = pt.get<double>("sim.sample_time", sample_time_);  

  // sim parameters
  seed_ = pt.get<int>("sim.seed", seed_);
  integration_step_ = pt.get<double>("sim.integration_step", integration_step_);
  signal_to_noise_  = pt.get<double>("sim.signal_to_noise", signal_to_noise_);

  gamma_1_ = pt.get<double>("sim.gamma_1", gamma_1_);
  gamma_2_ = pt.get<double>("sim.gamma_2", gamma_2_);
  gamma_g_ = pt.get<double>("sim.gamma_g", gamma_g_);

  larmor_freq_ = pt.get<double>("sim.larmor_freq", larmor_freq_);
  mixdown_freq_ = pt.get<double>("sim.mixdown_freq", mixdown_freq_);
  lowpass_ratio_ = pt.get<double>("sim.lowpass_ratio", lowpass_ratio_);
  mixdown_phi_ = pt.get<double>("sim.mixdown_phi", mixdown_phi_);

  pulse_freq_ = pt.get<double>("sim.pulse_freq", pulse_freq_);
  pulse_time_ = pt.get<double>("sim.pulse_time", pulse_time_);
  addnoise_ = pt.get<bool>("sim.addnoise", addnoise_);
  discrete_ = pt.get<bool>("sim.discrete", discrete_);

  amplitude_ = pt.get<double>("sim.amplitude", amplitude_);
  baseline_ = pt.get<double>("sim.baseline",baseline_);

  mixdown_freq_ = pt.get<double>("sim.mixdown_freq",mixdown_freq_);
  larmor_freq_ = pt.get<double>("sim.larmor_freq",larmor_freq_);
  lowpass_ratio_ = pt.get<double>("sim.lowpass_ratio",lowpass_ratio_);
  mixdown_phi_ = pt.get<double>("sim.mixdown_phi",mixdown_phi_);

  //spin_0_ = sim::spin_0;

  try {
    std::vector<double> tmp;
    for (auto &v : pt.get_child("grad.poln_coefs")){
      tmp.push_back(v.second.get_value<double>());
    }
    grad::poln_coefs = tmp;
  }
  catch (boost::property_tree::ptree_bad_path){};

}


// Create an idealized FID with current Simulation parameters
void FidFactory::IdealFid(std::vector<double>& wf, std::vector<double>& tm)
{
  wf.reserve(tm.size());
  wf.resize(0);

  // Define the waveform
  double temp;
  double w = dsp::kTau * (larmor_freq_ - mixdown_freq_);
  double phi = mixdown_phi_;
  double tau = 1.0 / gamma_1_;
  double amp = amplitude_;
  double base = baseline_;

  for (auto it = tm.begin(); it != tm.end(); ++it){

    if (*it >= pulse_time_){

      temp = amp * std::exp(-(*it - pulse_time_) / tau);
      temp *= std::sin((*it) * w + phi);
      wf.push_back(temp + base);

    } else {

      wf.push_back(base);

    }
  } 

  if (addnoise_) dsp::addwhitenoise(wf, signal_to_noise_);

  if (discrete_) dsp::floor(wf);
}


// Simulate an FID with the current class settings.
void FidFactory::SimulateFid(std::vector<double>& wf, std::vector<double>& tm)
{
  using namespace boost::numeric::odeint;
  using std::bind;
  using std::ref;
  namespace pl = std::placeholders; // _1, _2, _3

  // make sure memory is allocated for the final FIDs
  tm.reserve(num_samples_);
  wf.reserve(num_samples_);
  s_ = spin_0_; // The starting spin vector

  // Bind the member function and make a reference so it isn't copied.
  integrate_const(runge_kutta4<std::vector<double>>(), 
                  bind(&FidFactory::Bloch, ref(*this), pl::_1, pl::_2, pl::_3), 
                  s_, 
                  ti_, 
                  tf_,
                  dt_, 
                  bind(&FidFactory::Printer, ref(*this), pl::_1, pl::_2));

  // Set the results
  tm.resize(0);
  wf.resize(0);

  // Fill the FID vectors with simulation results.
  for (int i = 0; i < num_samples_; ++i) {
    tm.push_back(time_vec_[i * sim_to_fid_]);
    wf.push_back(spin_vec_[i * sim_to_fid_] + baseline_);
  }

  // Take care of optional effects.
  if (addnoise_) dsp::addwhitenoise(wf, signal_to_noise_);
  if (discrete_) dsp::floor(wf);
}


// Simulate a gradient FID by combining multiple simulated FIDs.
void FidFactory::GradientFid(const std::vector<double>& grad, 
                             std::vector<double>& wf)
{
  // Make sure we have an example file draw from.
  if (!pf_fid_->IsOpen()) {
    std::cout << "No ROOT file loaded.  Cannot make gradient FIDs.";
    std::cout << std::endl;
    return;
  }

  // Find the appropriate FIDs and sum them
  wf.assign(num_samples_, 0.0);


  // Add each FID in the gradient.
  for (auto& g : grad){

    pt_fid_->GetEntry(GetTreeIndex(g));

    for (uint i = 0; i < wf.size(); i++){
      wf[i] += wf_[i];
    }
  }

  // Calculate the proper amplitude_ and scale + offset the sim FID.
  double amp = amplitude_ / grad.size();

  for (auto& val : wf) {
    val = val * amp + baseline_;
  }

  // Take care of optional effects.
  if (addnoise_) dsp::addwhitenoise(wf, signal_to_noise_);
  if (discrete_) dsp::floor(wf);
}

void FidFactory::GradientFid_Bugao(std::vector<double>& wf, std::vector<double>& tm, std::vector<double>& fq)
{ 
  wf.assign(tm.size(), 0.0);

  int length = fq.size();
  double wf_temp;

  // Define the waveform
  double temp;
  double w;
  //double w = dsp::kTau * (larmor_freq_ - mixdown_freq_);
  double phi = mixdown_phi_;
  double tau = 1.0 / gamma_1_;
  double amp = amplitude_;
  double base = baseline_;

  for (int i=0;i<length;i++){
    w = dsp::kTau * fq[i];
    for (int j=0;j<tm.size();j++){

      if (tm[j] >= pulse_time_){

	temp = amp * std::exp(-(tm[j] - pulse_time_) / tau);
	temp *= std::sin((tm[j]) * w + phi);
	wf_temp = temp + base;

      } else {

	wf_temp = base;

      }

      wf[j] += wf_temp;

    }
  }
  for (int j=0;j<tm.size();j++){
    wf[j] /= length;
  }  

  if (addnoise_) dsp::addwhitenoise(wf, signal_to_noise_);

  if (discrete_) dsp::floor(wf);
} 


void FidFactory::PrintDiagnosticInfo()
{
  using std::cout;
  using std::endl;

  cout << endl;
  cout << "Printing Diagnostic Info for FidFactory @" << this << endl;
  cout << "The time step, fid length: " << dt_ << ", ";
  cout << num_samples_ << endl;
  cout << "The sim length, sim-to-fid: " << sim_length_ << ", ";
  cout << sim_to_fid_ << endl;
}


// Define the Bloch equation governing FID signals.
void FidFactory::Bloch(std::vector<double> const &s, 
                       std::vector<double> &dsdt, 
                       double t)
{
  // Again static to save time on memory allocations.
  static std::vector<double> bf = {{0., 0., 0.}};   // Bfield
  static std::vector<double> s1 = {{0., 0., 0.}};  // Cross product spin
  static std::vector<double> s2 = {{0., 0., 0.}};  // Relaxation spin

  // Update the Bfield.
  bf = Bfield(t);

  // Set the relaxtion bits of the differential.
  s2[0] = gamma_2_ * s[0];
  s2[1] = gamma_2_ * s[1];
  s2[2] = gamma_1_ * (s[2] - 1.0);

  // Calculate the cross product.
  dsp::cross(bf, s, s1);

  // Set the differential to be integrated.
  dsdt = dsp::VecSubtract(s1 , s2);
}


// A function that defines the static + RF B-fields.
std::vector<double> FidFactory::Bfield(const double& t)
{
  // Made static to save on memory calls.
  static std::vector<double> a = {0., 0., 0.}; // constant external field
  static std::vector<double> b = {0., 0., 0.}; // time dependent B field

  // Return static external field if after the pulsed field.
  if (t >= pulse_time_){
    return a;

  // Set the fields if the simulation is just starting.
  } else if (t <= ti_ + dt_) {

    a[2] = dsp::kTau * larmor_freq_;
    b[2] = dsp::kTau * larmor_freq_;

  }

  // Return static field if the time is before the pulsed field.
  if (t < 0.0) return a;

  // If none of the above, return the time-dependent, pulsed field.
  b[0] = dsp::kTau * pulse_freq_ * cos(dsp::kTau * mixdown_freq_ * t);
  b[1] = dsp::kTau * pulse_freq_ * sin(dsp::kTau * mixdown_freq_ * t);
  return b;
}

// The function called each step of integration.
void FidFactory::Printer(std::vector<double> const &s , double t)
{
  // Cache the cosine function for mixing later.
  if (cos_cache_.size() == 0){
    cos_cache_.reserve(sim_length_);

    double temp = t;
    double val;

    for (int i = 0; i < sim_length_; i++){
      val = cos(dsp::kTau * mixdown_freq_ * temp + mixdown_phi_);
      cos_cache_.push_back(val);
      temp += dt_;
    }

    spin_vec_.assign(sim_length_, 0.0);
    time_vec_.assign(sim_length_, 0.0);
  }

  // Make sure the index is reset
  if (t < ti_ + dt_) printer_idx_ = 0;

  // Record spin in the y-direction and mix down
  spin_vec_[printer_idx_] = amplitude_ * s[1] * cos_cache_[printer_idx_];

  // Record the time and increment printer_idx_
  time_vec_[printer_idx_++] = t;

  // If the FID is done, do more stuff.
  if (t > tf_ - dt_) {

    // Final spin sum from all different gradients
    spin_vec_ = LowPassFilter(spin_vec_);

    // Reset the counters
    printer_idx_ = 0; // They'll get incremented after.
  }
}

// Low pass filter suppresses higher frequencies from mixing down.
std::vector<double> FidFactory::LowPassFilter(std::vector<double>& s)
{
  // Allocate the filter and set the central frequency.
  int zero_padding = 0;
  int N = sim_length_ + 2 * zero_padding;
  double freq_cut = lowpass_ratio_ * larmor_freq_;

  std::vector<double> tmp(N);
  std::vector<double> filter(N / 2 + 1);

  // Define the filter using 3rd order Butterworth filter.
  unsigned int i = 0;
//  int j = N - 1;
  double temp;

  // The filter is symmetric, so we can fill both sides in tandem.
  while (i < filter.size()){
    // scaling term
    temp = pow(i / (dt_ * N * freq_cut), 8);
    filter[i++] = pow(1.0 / (1.0 + temp), 0.5);
  }

  // Copy the spin vector to a temporary vector.
  for (int i = 0; i < N; ++i) {
    if (i < zero_padding) {
      tmp[i] = 0.0;
    } else if (i > N - 2 * zero_padding) {
      tmp[i] = 0.0;
    } else {
      tmp[i] = s[i - zero_padding];
    }
  }

  auto fft = dsp::rfft(tmp);

  for (unsigned int i = 0; i < fft.size(); ++i) {
    fft[i] *= filter[i];
  }

  auto rv = dsp::irfft(fft, sim_length_ % 2 == 1);

  // Copy the vectors to Armadillo formats first.
  //arma::vec v1(&tmp[0], tmp.size(), false);
  //arma::vec v2(&filter[0], filter.size(), false);

  // Now get the FFT
  // arma::cx_vec fft = arma::fft(v1);

  // Multiply the FFT and the filter element wise
  // fft = fft % v2;

  // This confusing oneliner, get the inverse FFT and converts to a std::vector
  // auto rv = arma::conv_to<std::vector<double>>::from(arma::real(arma::ifft(fft)));

  for (unsigned int i = 0; i < s.size(); ++i) {
    s[i] = rv[i + zero_padding];
  }

  return s;
}

int FidFactory::GetTreeIndex(double grad_strength)
{
  return (int)(std::nearbyint(grad_strength / d_grad_ + zero_idx_) + 0.5);
}

} // ::fid
