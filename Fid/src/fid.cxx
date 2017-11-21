#include "fid.h"
#include <chrono>

namespace fid {

Fid::Fid()
{
  NBatch = 0;
  fid_size = 0;
  FFTDone = false;
  for (int i=0 ; i<8 ; i++){
    freq_array_[i] = 0;
    freq_err_array_[i] = 0;
  }
}

Fid::Fid(const std::vector<double>& wf, const std::vector<double>& tm)
{
  NBatch = 1;
  fid_size = wf.size();
  FFTDone = false;
  for (int i=0 ; i<8 ; i++){
    freq_array_[i] = 0;
    freq_err_array_[i] = 0;
  }
  // Copy the waveform and time to member vectors.
  wf_ = wf;
  tm_ = tm;
}

Fid::Fid(const std::vector<double>& wf, double dt, unsigned int tNBatch, unsigned int tSize)
{
  NBatch = tNBatch;
  fid_size = tSize;;
  FFTDone = false;
  if(wf.size() != NBatch*fid_size){
    std::cout << "Warning: waveform vector size does not match the given arguments."<<std::endl;
  }
  for (int i=0 ; i<8 ; i++){
    freq_array_[i] = 0;
    freq_err_array_[i] = 0;
  }
  // Copy the waveform and time to member vectors.
  wf_ = wf;
  tm_.resize(wf_.size());
  for (unsigned int i;i<NBatch;i++){
    for (unsigned int j;j<fid_size;j++){
      tm_[i*fid_size+j] = j*dt;
    }
  }
}

Fid::SetWf(const std::vector<double>& wf, double dt, unsigned int tNBatch, unsigned int tSize)
{
  //Clear the existing wf
  wf_.clear();
  tm_.clear(); //time stamps
  filtered_wf_.clear(); // filtered waveform
  baseline_.clear(); // baseline
  fid_fft_.clear(); //spectrum after fft
  wf_im_.clear(); //harmonic conjugate of the waveform
  psd_.clear(); //power density
  env_.clear(); //envelope
  phi_.clear(); //monotonic phase
  fftfreq_.clear(); // fft frequency stamp
  res_.clear(); //residual of fit
  //Renew parameters
  NBatch = tNBatch;
  fid_size = tSize;;
  FFTDone = false;
  if(wf.size() != NBatch*fid_size){
    std::cout << "Warning: waveform vector size does not match the given arguments."<<std::endl;
  }
  for (int i=0 ; i<8 ; i++){
    freq_array_[i] = 0;
    freq_err_array_[i] = 0;
  }
  // Copy the waveform and time to member vectors.
  wf_ = wf;
  tm_.resize(wf_.size());
  for (unsigned int i;i<NBatch;i++){
    for (unsigned int j;j<fid_size;j++){
      tm_[i*fid_size+j] = j*dt;
    }
  }
}

Fid::Fid(const std::vector<double>& wf)
{
  NBatch = 1;
  fid_size = wf.size();
  FFTDone = false;
  for (int i=0 ; i<8 ; i++){
    freq_array_[i] = 0;
    freq_err_array_[i] = 0;
  }
  // Copy the waveform and construct a generic time range.
  wf_ = wf;
  tm_ = dsp::construct_range(0.0, (double)wf_.size(), 1.0);
}

void Fid::SetParameter(const std::string& Name, double Value)
{
  if (Name.compare("edge_width")==0){
    edge_width_ = Value;
  }else if (Name.compare("edge_ignore")==0){ 
    edge_ignore_ = Value;
  }else if (Name.compare("start_amplitude")==0){ 
    start_amplitude_ = Value;
  }else if (Name.compare("baseline_freq_thresh")==0){ 
    baseline_freq_thresh_ = Value;
  }else if (Name.compare("filter_low_freq")==0){ 
    filter_low_freq_ = Value;
  }else if (Name.compare("filter_high_freq")==0){ 
    filter_high_freq_ = Value;
  }else if (Name.compare("fft_peak_width")==0){ 
    fft_peak_width_ = Value;
  }else if (Name.compare("centroid_thresh")==0){ 
    centroid_thresh_ = Value;
  }else if (Name.compare("hyst_thresh")==0){ 
    hyst_thresh_ = Value;
  }else if (Name.compare("snr_thresh")==0){ 
    snr_thresh_ = Value;
  }else if (Name.compare("len_thresh")==0){ 
    len_thresh_ = Value;
  }
}

void Fid::SetNoise(double NoiseValue){
  noise_ = NoiseValue; 
}

void Fid::SetBaseline(const std::vector<double>& bl){
  baseline_ = bl;
}

void Fid::Init(std::string Option)
{
  // Prevent memory issues with the TF1s in root6.
#ifndef __ROOTCLING__
  TF1::DefaultAddToGlobalList(false);
#endif

  // Initialize the health properly.
  health_ = 100.0;

  if (Option.compare("Speed")==0){
  //Test Time cost
  auto t0 = std::chrono::high_resolution_clock::now();
    // Initialize the Fid for analysis
    CenterFid();
    CalcNoise();
    CalcMaxAmp();
    if (noise_>0.0){
      snr_ = max_amp_ / noise_;
    }else{
      snr_ = 1e7; // If noise is not set, snr is set to very high
    }
    FindFidRange();

    freq_method_ = ZC;
    //Calculate Frequncy
    CalcFreq();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
  double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
  //std::cout << "Time = "<<t<<std::endl;
//  std::cout << "Freq = "<<freq_array_[ZC]<<std::endl;
//  std::cout << "EdgeWidth = "<<edge_width_<<std::endl;
  //std::cout << "Range = "<<tm_[i_wf_] <<" "<<tm_[f_wf_]<<std::endl;
  }else if (Option.compare("Standard")==0){
    CallFFT();
    CalcPowerEnvAndPhase();
    BaselineCorrection();
    CalcMaxAmp();
    FindFidRange();
    GuessFitParams();
    CalcNoise();
    if (noise_>0.0){
      snr_ = max_amp_ / noise_;
    }else{
      snr_ = 1e7;// If noise is not set, snr is set to very high
    }

    freq_method_ = PH;
    CalcFreq();
  }else if (Option.compare("FFTOnly")==0){
    CallFFT();
    CalcPowerEnvAndPhase();
  }else if (Option.compare("Old")==0){
    CenterFid();
    CalcNoise();
    CalcMaxAmp();
    if (noise_>0.0){
      snr_ = max_amp_ / noise_;
    }else{
      snr_ = 1e7;// If noise is not set, snr is set to very high
    }
    FindFidRange();

    //Get Frequency vector
    CalcFftFreq();
    // Get the fft of the waveform first.
    fid_fft_ = dsp::rfft(wf_);

    //Filtered fft spectrum
    auto fid_fft_filtered_ = dsp::window_filter(fid_fft_,fftfreq_,filter_low_freq_,filter_high_freq_);

    // Apply square filter and reverse fft to get the filtered waveform
    filtered_wf_ = dsp::irfft(fid_fft_filtered_, wf_.size() % 2 == 1);

    // Now get the imaginary harmonic complement to the waveform.
    wf_im_ = dsp::hilbert(wf_);

    // Get the baseline by applying low-pass filter
    baseline_ = dsp::irfft(dsp::window_filter(fid_fft_,fftfreq_,0,baseline_freq_thresh_), wf_.size() % 2 == 1);

    psd_ = dsp::norm(fid_fft_);
    phi_ = dsp::phase(wf_, wf_im_);
    env_ = dsp::envelope(wf_, wf_im_);

    // find max index and set the fit window
    // Must rule out the constant component psd_[0]
    max_idx_fft_ = std::distance(psd_.begin(),
	std::max_element(psd_.begin()+1, psd_.end()));

    double interval = fftfreq_[1] - fftfreq_[0];
    unsigned int fft_peak_index_width = static_cast<int>(fft_peak_width_/interval);

    if (max_idx_fft_ < fft_peak_index_width) i_fft_ = 1;  
    else i_fft_ = max_idx_fft_ - fft_peak_index_width ;

    f_fft_ = max_idx_fft_ + fft_peak_index_width;
    if (f_fft_ > psd_.size()) f_fft_ = psd_.size();

    FFTDone = true;
    GuessFitParams();

    freq_method_ = PH;
    CalcFreq();
  }

  // Flag the FID as bad if it's negative.
  if (freq_array_[freq_method_] <= 0.0) {
    health_ = 0.0;
  }

  // Else calculate a health based on signal to noise.
  /*if (max_amp_ < noise_ * snr_thresh_) {
    health_ *= max_amp_ / (noise_ * snr_thresh_);
  }*/
  if (snr_ < snr_thresh_) {
    health_ *= (snr_ / snr_thresh_);
  }

  // And factor fid duration into health.
  if (f_wf_ - i_wf_ < wf_.size() * len_thresh_) {
    health_ *= (f_wf_ - i_wf_) / (wf_.size() * len_thresh_);
  }

}


void Fid::CenterFid()
{
  int w = static_cast<int>(edge_width_/(tm_[1]-tm_[0]));
  double sum  = std::accumulate(wf_.begin(), wf_.begin() + w, 0.0);
  double avg = sum / w; // to pass into lambda
  mean_ = avg; // save to class
  //std::cout <<"Baseline " << avg <<std::endl;

  std::for_each(wf_.begin(), wf_.end(), [avg](double& x){ x -= avg; });
}

void Fid::BaselineCorrection()
{
  if (baseline_.size() == wf_.size() ){
    dsp::VecSubtract(wf_ , baseline_); 
  }
}

void Fid::ConstBaselineCorrection(double ConstBaseline){
  dsp::VecShift(-ConstBaseline, wf_);
}

void Fid::CalcNoise()
{
  // Grab a new handle to the noise window width for aesthetics.
  int i = static_cast<int>(edge_ignore_/(tm_[1]-tm_[0]));
  int f = static_cast<int>(edge_width_/(tm_[1]-tm_[0])) + i;

  // Find the noise level in the head and tail.
  double head = dsp::stdev(wf_.begin() + i, wf_.begin() + f);
  double tail = dsp::stdev(wf_.rbegin() + i, wf_.rbegin() + f);

  // Take the smaller of the two.
  noise_ = (tail < head) ? (tail) : (head);
}

double Fid::CalcRms()
{
  int ignore = static_cast<int>(edge_ignore_/(tm_[1]-tm_[0]));

  // Find the rms of the entire range, excluding the edge_ignore
  return dsp::stdev(wf_.begin() + ignore, wf_.end() - ignore);
}

void Fid::CalcMaxAmp()
{
  //Waveform select
  std::vector<double>* wfptr;
  if (FFTDone){
    wfptr = &filtered_wf_;
  }else{
    wfptr = &wf_;
  }
  auto mm = std::minmax_element(wfptr->begin(), wfptr->end());

  if (std::abs(*mm.first) > std::abs(*mm.second)) {

    max_amp_ = std::abs(*mm.first);

  } else {

    max_amp_ = std::abs(*mm.second);
  }
}


void Fid::FindFidRange()
{
  // Find the starting and ending points
  double thresh = start_amplitude_ * max_amp_;
  bool checks_out = false;

  //Waveform select
  std::vector<double>* wfptr;
  if (FFTDone){
    wfptr = &filtered_wf_;
  }else{
    wfptr = &wf_;
  }

  // Find the first element with magnitude larger than thresh
  int IgnoreRange = static_cast<int>(edge_ignore_/(tm_[1]-tm_[0]));
  auto it_1 = wfptr->begin() + IgnoreRange;

  while (!checks_out) {

    // Check if the point is above threshold.
    auto it_i = std::find_if(it_1, wfptr->end(),
      [thresh](double x) {
        return std::abs(x) > thresh;
    });

    // Make sure the point is not with one of the vector's end.
    if ((it_i != wfptr->end()) && (it_i + 1 != wfptr->end())) {

      // Check if the next point is also over threshold.
      checks_out = std::abs(*(it_i + 1)) > thresh;

      // Increase the comparison starting point.
      it_1 = it_i + 1;

      // Turn the iterator into an index
      if (checks_out) {
        i_wf_ = std::distance(wfptr->begin(), it_i);
      }

    } else {

      // If we have reached the end, mark it as the last.
      i_wf_ = std::distance(wfptr->begin(), wfptr->end());
      break;
    }
  }

  // Find the next element with magnitude lower than thresh
  auto it_2 = std::find_if(wfptr->begin() + i_wf_, wfptr->end(),
      [thresh](double x) {
        return std::abs(x) < 0.8 * thresh;
  });

  if ((it_2 != wfptr->end()) && (it_2 + 1 != wfptr->end())) {

    checks_out = false;

  } else {

    f_wf_ = std::distance(wfptr->begin(), wfptr->end());
    checks_out = true;
  }

  while (!checks_out) {

    // Find the range around a peak.
    auto it_i = std::find_if(it_2, wfptr->end(),
      [thresh](double x) {
        return std::abs(x) > 0.8 * thresh;
    });

    auto it_f = std::find_if(it_i + 1, wfptr->end(),
      [thresh](double x) {
        return std::abs(x) < 0.8 * thresh;
    });

    // Now check if the peak actually made it over threshold.
    if ((it_i != wfptr->end()) && (it_f != wfptr->end())) {

      auto mm = std::minmax_element(it_i, it_f);

      if ((*mm.first < -thresh) || (*mm.second > thresh)) {

        it_2 = it_f;

      } else {

        checks_out = true;
      }

      // Turn the iterator into an index
      if (checks_out) {
        f_wf_ = std::distance(wfptr->begin(), it_f);
      }

    } else {

      f_wf_ = std::distance(wfptr->begin(), wfptr->end());
      break;
    }
  }

  if (f_wf_ > wfptr->size()-IgnoreRange){
    f_wf_ = wfptr->size()-IgnoreRange;
  }

  // Gradients can cause a waist in the amplitude.
  // Mark the signal as bad if it didn't find signal above threshold.
  if (i_wf_ > wfptr->size() * 0.95 || i_wf_ >= f_wf_) {

    health_ = 0.0;

    i_wf_ = 0;
    f_wf_ = wfptr->size() * 0.01;
  }
}

double Fid::GetFreq(const std::string& method_name)
{
  return GetFreq(ParseMethod(method_name));
}

double Fid::GetFreq(const Method m)
{
  // Recalculate if necessary
  if (freq_array_[m]<=0) {
    freq_method_ = m;
    CalcFreq();
  } 

  return freq_array_[m];
}

double Fid::GetFreqError(const std::string& method_name)
{
  return GetFreqError(ParseMethod(method_name));
}

double Fid::GetFreqError(const Method m)
{
  // Recalculate if necessary
  if (freq_err_array_[m]<=0) {
    freq_method_ = m;
    CalcFreq();
  } 

  return freq_err_array_[m];
}

// Calculate the frequency using the current Method
void Fid::CalcFreq()
{
  switch(freq_method_) {

    case ZC:
    case ZEROCOUNT:
      CalcZeroCountFreq();
      break;

    case CN:
    case CENTROID:
      CalcCentroidFreq();
      break;

    case AN:
    case ANALYTICAL:
      CalcAnalyticalFreq();
      break;

    case LZ:
    case LORENTZIAN:
      CalcLorentzianFreq();
      break;

    case EX:
    case EXPONENTIAL:
      CalcExponentialFreq();
      break;

    case PH:
    case PHASE:
      CalcPhaseFreq();
      break;

    case SN:
    case SINUSOID:
      CalcSinusoidFreq();
      break;

    case PD:
    case PHASEDERIVITIVE:
      CalcPhaseDerivFreq();
      break;

    default:
      // Reset the Method to a valid one.
      freq_method_ = PH;
      CalcPhaseFreq();
      break;
  }
}

void Fid::CallFFT()
{
  //Get Frequency vector
  CalcFftFreq();
  // Get the fft of the waveform first.
  fid_fft_ = dsp::rfft(wf_);

  //Filtered fft spectrum
  auto fid_fft_filtered_ = dsp::window_filter(fid_fft_,fftfreq_,filter_low_freq_,filter_high_freq_);

  // Apply square filter and reverse fft to get the filtered waveform
  filtered_wf_ = dsp::irfft(fid_fft_filtered_, wf_.size() % 2 == 1);

  // Now get the imaginary harmonic complement to the waveform.
  for (auto it = fid_fft_filtered_.begin() + 1; it != fid_fft_filtered_.end(); ++it) {
    *it = dsp::cdouble((*it).imag(), -(*it).real());
  }

  //wf_im_ = dsp::irfft(fid_fft_filtered_, wf_.size() % 2 == 1);
  wf_im_ = dsp::irfft(fid_fft_filtered_, wf_.size() % 2 == 1);

  // Get the baseline by applying low-pass filter
  auto baseline_fft = dsp::window_filter(fid_fft_,fftfreq_,0,baseline_freq_thresh_);
  //std::vector<dsp::cdouble> baseline_fft(fid_fft_.size());
  //baseline_fft[0] = fid_fft_[0]-fid_fft_[1];
  baseline_ = dsp::irfft(baseline_fft, wf_.size() % 2 == 1);

  FFTDone = true;
}

void Fid::CalcPowerEnvAndPhase()
{
  // Now we can get power, envelope and phase.
  psd_ = dsp::norm(fid_fft_);
  phi_ = dsp::phase(filtered_wf_, wf_im_);
  env_ = dsp::envelope(filtered_wf_, wf_im_);

  // find max index and set the fit window
  // Must rule out the constant component psd_[0]
  max_idx_fft_ = std::distance(psd_.begin(),
    std::max_element(psd_.begin()+1, psd_.end()));

  double interval = fftfreq_[1] - fftfreq_[0];
  unsigned int fft_peak_index_width = static_cast<int>(fft_peak_width_/interval);

  if (max_idx_fft_ < fft_peak_index_width) i_fft_ = 1;  
  else i_fft_ = max_idx_fft_ - fft_peak_index_width ;

  f_fft_ = max_idx_fft_ + fft_peak_index_width;
  if (f_fft_ > psd_.size()) f_fft_ = psd_.size();
}

void Fid::CalcFftFreq()
{
  // @todo: consider storing as start, step, stop
  fftfreq_ = dsp::fftfreq(tm_);
}

void Fid::GuessFitParams()
{
  // Guess the general fit parameters
  guess_.assign(6, 0.0);

  double f_mean;
  double f_mean2;
  double den;

  auto it_pi = psd_.begin() + i_fft_; // to shorten subsequent lines
  auto it_pf = psd_.begin() + f_fft_;
  auto it_fi = fftfreq_.begin() + i_fft_;

  // Compute some moments
  f_mean = std::inner_product(it_pi, it_pf, it_fi, 0.0);
  den = std::accumulate(it_pi, it_pf, 0.0); // normalization
  f_mean /= den;

  // find average power squared
  f_mean2 = std::inner_product(it_pi, it_pf, it_fi, 0.0,
    [](double sum, double x) {return sum + x;},
    [](double x1, double x2) {return x1 * x2 * x2;});
  f_mean2 /= den;

  // frequency
  guess_[0] = f_mean;

  // peak width
  guess_[1] = std::sqrt(f_mean2 - f_mean * f_mean);

  // amplitude
  guess_[2] = psd_[max_idx_fft_];

  // background
  //guess_[3] = noise_;
  guess_[3] = psd_[i_fft_];

  // exponent
  guess_[4] = 2.0;

  return;
}


void Fid::FreqFit(TF1& func)
{
  // Make a TGraph to fit
  gr_freq_series_ = TGraph(f_fft_ - i_fft_, &fftfreq_[i_fft_], &psd_[i_fft_]);

  gr_freq_series_.Fit(&func, "QMRSEX0", "C", fftfreq_[i_fft_], fftfreq_[f_fft_]);
  //gr_freq_series_.Fit(&func, "MRSEX0", "C", fftfreq_[i_fft_], fftfreq_[f_fft_]);

  chi2_ = func.GetChisquare();
  freq_err_array_[freq_method_] = f_fit_.GetParError(0) / dsp::kTau;

  res_.resize(0);
  for (unsigned int i = i_fft_; i < f_fft_ + 1; ++i){
    res_.push_back(psd_[i] - func.Eval(fftfreq_[i]));
  }

  return;
}

double Fid::CalcZeroCountFreq()
{
  std::vector<double> temp_;
  // set up vectors to hold relevant stuff about the important part
  temp_.resize(f_wf_ - i_wf_);

  // printf("fid range: %i, %i\n", i_wf_, f_wf_);

  int nzeros = 0;
  bool pos = wf_[i_wf_] >= 0.0;
  bool hyst = false;
  bool skip_first = true;
  
  auto mm = std::minmax_element(wf_.begin(), wf_.end()); // returns {&min, &max}

  double max = *mm.second;
  if (std::abs(*mm.first) > max) max = std::abs(*mm.first);
  
  // double max = (-(*mm.first) > *mm.second) ? -(*mm.first) : *mm.second;
  double thresh = hyst_thresh_ * max;
  //  thresh = 10 * noise_;

  int i_zero = -1;
  int f_zero = -1;

  // printf("env size: %i\n", env_.size());

  // iterate over vector
  for (unsigned int i = i_wf_; i < f_wf_; i++){

    // hysteresis check
    if (hyst){
      hyst = std::abs(wf_[i]) < thresh;
      continue;
    }

    // check for a sign change
    if ((wf_[i] >= 0.0) != pos){

      if (skip_first) {

        skip_first = false;
        pos = !pos;
        hyst = true;

      } else {

        nzeros++;
        f_zero = i;
        if (i_zero == -1) i_zero = i;
        pos = !pos;
        hyst = true;
      }
    }
  }

  // printf("finished looking for zeros\n");

  // Use linear interpolation to find the zeros more accurately
  int i = i_zero;
  int f = f_zero;

  // do the interpolation
  double frac = std::abs(wf_[i] / (wf_[i-1] - wf_[i]));
  double ti = frac * tm_[i-1] + (1.0 - frac) * tm_[i];

  frac = std::abs(wf_[f] / (wf_[f-1] - wf_[f]));
  double tf = frac * tm_[f-1] + (1.0 - frac) * tm_[f];

  freq_array_[ZC] = 0.5 * (nzeros - 1.0) / (tf - ti);
  // todo: Fix this into a better error estimate. 
  freq_err_array_[ZC] = freq_array_[ZC] * sqrt(2) * (tm_[1] - tm_[0]) / (tf - ti);

  return freq_array_[ZC];
}

double Fid::CalcCentroidFreq()
{
  // Find the peak power
  double thresh = *std::max_element(psd_.begin(), psd_.end());
  thresh *= centroid_thresh_;

  // Find the indices for a window around the max
  int it_i = std::distance(psd_.begin(), 
    std::find_if(psd_.begin(), psd_.end(), 
      [thresh](double x) {return x > thresh;}));

  // reverse the iterators
  int it_f = -1 * std::distance(psd_.rend(),
    std::find_if(psd_.rbegin(), psd_.rend(), 
      [thresh](double x) {return x > thresh;}));

  // Now compute the power weighted average
  double pwfreq = 0.0;
  double pwfreq2 = 0.0;
  double pwsum = 0.0;

  for (int i = it_i; i < it_f; i++){
    pwfreq += psd_[i] * fftfreq_[i];
    pwfreq2 += psd_[i] * psd_[i] * fftfreq_[i];
    pwsum  += psd_[i];
  }

  freq_err_array_[CN] = sqrt(pwfreq2 / pwsum - pow(pwfreq / pwsum, 2.0));
  freq_array_[CN] = pwfreq / pwsum;

  return freq_array_[CN];
}


double Fid::CalcAnalyticalFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  // Set the fit function:
  // p0 -> fid frequency
  // p1 -> decay gamma
  // p2 -> amplitude
  // p3 -> baseline offset
  // p4 -> phi
  std::string fcn("[3]  + [2] * (([0] * cos([4]) + [1] * sin([4]))^2 ");
  fcn += std::string("+ (x * sin([1]))^2) / (([1]^2 + [0]^2 - x^2)^2 ");
  fcn += std::string("+ 4 * x^2 * [1]^2)");

  f_fit_ = TF1("f_fit_", fcn.c_str(), fftfreq_[i_fft_], fftfreq_[f_fft_]);

  // Set the parameter guesses
  for (unsigned int i = 0; i < 5; i++){
    f_fit_.SetParameter(i, guess_[i]);
  }

  // Limits
  f_fit_.SetParLimits(4, 0.0, dsp::kTau);

  FreqFit(f_fit_);

  freq_array_[AN] = f_fit_.GetParameter(0);

  return freq_array_[AN];
}

/*
double Fid::CalcLorentzianFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  // Set the fit function
  std::string fcn("[2] / (1 + ((x - [0]) / (0.5 * [1]))^2) + [3]");
  f_fit_ = TF1("f_fit_", fcn.c_str(), fftfreq_[i_fft_], fftfreq_[f_fft_]);
  f_fit_.SetParLimits(3,guess_[3]*0.8,guess_[3]*1.2);

  // Set the parameter guesses
  for (int i = 0; i < 4; i++){
    f_fit_.SetParameter(i, guess_[i]);
  }

  FreqFit(f_fit_);

  freq_array_[LZ] = f_fit_.GetParameter(0);

  return freq_array_[LZ];
}*/

double Fid::CalcLorentzianFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  //Lorentzian function
  //[2] / (1 + ((x - [0]) / (0.5 * [1]))^2) + [3]
  //calculate[3]
  std::vector<double> parameter(3);
  double bkg = 0;
  for(int i=i_fft_-50 ; i-i_fft_<0 ; i++)
  {
    if(i>0)
    {
      bkg=bkg+psd_[i];
    }
    else 
    {
      ;
    }
  }
  bkg/=50.0;
  //inverse the data
  for(int i=i_fft_ ; i-f_fft_<=0 ; i++)
  {
    psd_[i]=1/(psd_[i]-bkg);
  }
  //use linear fit
  dsp::linear_fit(fftfreq_,psd_,i_fft_,f_fft_,3,parameter,res_);
  //change the parameter back
  freq_array_[LZ] = -parameter[1]*0.5/parameter[2];
  return freq_array_[LZ];
}

double Fid::CalcSoftLorentzianFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  // Set the fit function
  f_fit_ = TF1("f_fit_", "[2] / (1 + ((x - [0]) / (0.5 * [1]))^[4]) + [3]");

  // Set the parameter guesses
  for (int i = 0; i < 5; i++){
    f_fit_.SetParameter(i, guess_[i]);
  }

  f_fit_.SetParLimits(4, 1.0, 3.0);

  FreqFit(f_fit_);

  freq_array_[LZ] = f_fit_.GetParameter(0);

  return freq_array_[LZ];
}


double Fid::CalcExponentialFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  // Set the fit function
  f_fit_ = TF1("f_fit_", "[2] * exp(-abs(x - [0]) / [1]) + [3]");

  // Set the parameter guesses
  for (int i = 0; i < 4; i++){
    f_fit_.SetParameter(i, guess_[i]);
  }

  f_fit_.SetParameter(1, 0.5 * guess_[1] / std::log(2));

  FreqFit(f_fit_);

  freq_array_[EX] = f_fit_.GetParameter(0);

  return freq_array_[EX];
}


double Fid::CalcPhaseFreq()
{
  int poln = 1;
  //Requires FFT
  if (!FFTDone)return 0;
//  gr_time_series_ = TGraph(f_wf_ - i_wf_, &tm_[i_wf_], &phi_[i_wf_]);

  // Now set up the polynomial phase fit
  char fcn[20];
  sprintf(fcn, "pol%d", poln);
  f_fit_ = TF1("f_fit_", fcn,tm_[i_wf_],tm_[f_wf_]);

  // Adjust to ignore the edges if possible.
/*  int edge = static_cast<int>(edge_ignore_/(tm_[1]-tm_[0]));
  if (f - i > 2 * edge) {
    i += edge;
    f -= edge;
  }*/

  // Do the fit.
  //TEST linear fit
  /*
  std::vector<double> x= {0,1,2,3};
  std::vector<double> y= {0,2,4,6};
  dsp::linear_fit(x,y,0,4,2,ParList);
  */
  std::vector<double> ParList(poln+1);
  dsp::linear_fit(tm_,phi_,i_wf_,f_wf_,poln+1,ParList,res_);
  for (int i=0;i<=poln;i++){
    f_fit_.SetParameter(i,ParList[i]);
  }

  freq_array_[PH] = ParList[1] / dsp::kTau;
//  freq_err_array_[PH] = f_fit_.GetParError(1) / dsp::kTau;
  chi2_ = dsp::VecChi2(res_);

//  std::cout <<"Linear Fit Result: " << ParList[1]/dsp::kTau<<std::endl;

  return freq_array_[PH];
}

double Fid::CalcPhaseDerivFreq(int poln)
{
  //Requires FFT
  if (!FFTDone)return 0;

  // Now set up the polynomial phase fit
  char fcn[20];
  sprintf(fcn, "pol%d", poln);
  f_fit_ = TF1("f_fit_", fcn,tm_[i_wf_],tm_[f_wf_-1]);

  //Fit
  std::vector<double> ParList(poln+1);
  dsp::linear_fit(tm_,phi_,i_wf_,f_wf_,poln+1,ParList,res_);
  for (int i=0;i<=poln;i++){
    f_fit_.SetParameter(i,ParList[i]);
  }

//  freq_err_array_[PH] = f_fit_.GetParError(1) / dsp::kTau;
  chi2_ = dsp::VecChi2(res_);

//  std::cout <<"Linear Fit Result: " << ParList[1]/dsp::kTau<<std::endl;

  // Find the initial phase by looking at the function's derivative
  freq_array_[PD] = f_fit_.Derivative(tm_[i_wf_]) / dsp::kTau;

  return freq_array_[PD];
} 

/*
double Fid::CalcSinusoidFreq()
{
  //Requires FFT
  if (!FFTDone)return 0;
  std::vector<double> temp_;
  // Normalize the waveform by the envelope
  temp_.resize(wf_.size());

  std::transform(wf_.begin(), wf_.end(), env_.begin(), temp_.begin(),
    [](double x_wf, double x_env) {return x_wf / x_env;});

  gr_time_series_ = TGraph(f_wf_ - i_wf_, &tm_[i_wf_], &temp_[i_wf_]);    

  f_fit_ = TF1("f_fit_", "[1] * sin([0] * x) + [2] * cos([0] * x)");

  // Guess parameters
  f_fit_.SetParameter(0, dsp::kTau * CalcZeroCountFreq());
  f_fit_.SetParameter(1, 0.71);
  f_fit_.SetParameter(2, 0.71);
  f_fit_.SetParLimits(1, -1.1, 1.1);
  f_fit_.SetParLimits(2, -1.1, 1.1);

  // Adjust to ignore the edges
  int edge = static_cast<int>(edge_ignore_/(tm_[1]-tm_[0]));
  int i = i_wf_ + edge;
  int f = f_wf_ - edge;

  // Do the fit.
 gr_time_series_.Fit(&f_fit_, "QMRSEX0", "C", tm_[i], tm_[f]);

  res_.resize(0);
  for (unsigned int i = i_wf_; i < f_wf_ + 1; ++i){
    res_.push_back(temp_[i] - f_fit_.Eval(tm_[i]));
  }

  freq_array_[SN] = f_fit_.GetParameter(0) / dsp::kTau;
  ireq_err_array_[SN] = f_fit_.GetParError(0) / dsp::kTau;
  chi2_ = f_fit_.GetChisquare();

  return freq_array_[SN];
}
*/

Method Fid::ParseMethod(const std::string& m)
{
  using std::string;

  // Test each case iteratively, Zero count first.
  string str1("ZEROCOUNT");
  string str2("ZC");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::ZC;
  }

  str1 = string("CENTROID");
  str2 = string("CN");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::CN;
  }

  str1 = string("ANALYTICAL");
  str2 = string("AN");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::AN;
  }

  str1 = string("LORENTZIAN");
  str2 = string("LZ");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::LZ;
  }

  str1 = string("EXPONENTIAL");
  str2 = string("EX");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::EX;
  }

  str1 = string("PHASE");
  str2 = string("PH");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::PH;
  }

  str1 = string("SINUSOID");
  str2 = string("SN");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::SN;
  }

  str1 = string("PHASEDERIVITIVE");
  str2 = string("PD");

  if (m.compare(str1)==0 || m.compare(str2)==0){
    return Method::PD;
  }

  // If the method hasn't matched yet, use the current method and warn them.
  std::cout << "Warning: Method string not matched. " << std::endl;
  std::cout << "Method not changed." << std::endl;

  return freq_method_;
}

} // fid
