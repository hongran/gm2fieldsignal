#include "dsp.h"
#include <cassert>

namespace dsp {

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

std::vector<double> hilbert(const std::vector<double>& v)
{
	// Return the call to the fft version.
	auto fft_vec = rfft(v);

  // Zero out the constant term.
  fft_vec[0] = cdouble(0.0, 0.0);

  // Multiply in the -i.
  for (auto it = fft_vec.begin() + 1; it != fft_vec.end(); ++it) {
    *it = cdouble((*it).imag(), -(*it).real());
  }


  // Reverse the fft.
  return irfft(fft_vec, v.size() % 2 == 1);
}

std::vector<double> psd(const std::vector<double>& v)
{
  // Perform fft on the original data.
	auto fft_vec = rfft(v);

  // Get the norm of the fft as that is the power.
	return norm(fft_vec);
}

// Helper function to get frequencies for FFT
std::vector<double> fftfreq(const std::vector<double>& tm) 
{
	int N = tm.size();
	double dt = (tm[N-1] - tm[0]) / (N - 1); // sampling rate

	return fftfreq(N, dt);
}

std::vector<double> fftfreq(const int N, const double dt)
{
	// Instantiate return vector.
	std::vector<double> freq;

	// Handle both even and odd cases properly.
	if (N % 2 == 0) {

		freq.resize(N/2 + 1);
		
		for (unsigned int i = 0; i < freq.size(); ++i) {
			freq[i] = i / (dt * N);
		}

	} else {

		freq.resize((N + 1) / 2);

		for (unsigned int i = 0; i < freq.size(); ++i){
			freq[i] = i / (dt * N);
		}
	}

	return freq;
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
