#include "userfid.h"
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace fid {

// Load FID data from a formatted text file.
void UserFid::LoadTextData(std::string filename)
{
  // open the file first
  std::ifstream in(filename);

  // shrink vectors
  wf_.resize(0);
  tm_.resize(0);

  double wf_temp;
  double tm_temp;

  while (in.good()) {
    in >> tm_temp >> wf_temp;
    tm_.push_back(tm_temp);
    wf_.push_back(wf_temp);
  }
}

// Load all the current parameters in the fid::params namespace.
void UserFid::LoadParams(std::string conf_file)
{
  // using directives
  using boost::property_tree::ptree;

  // Declare and load JSON params.
  ptree pt;
  read_json(conf_file, pt);
  pt = pt.get_child("fid");

  // analysis parameters
  fft_peak_width_ = pt.get<int>("params.fft_peak_width", fft_peak_width_);
  edge_ignore_ = pt.get<double>("params.edge_ignore", edge_ignore_);
  edge_width_ = pt.get<double>("params.edge_width", edge_width_);
  start_amplitude_ = pt.get<double>("params.start_amplitude", start_amplitude_);
  centroid_thresh_ = pt.get<double>("params.centroid_thresh", centroid_thresh_);
  hyst_thresh_ = pt.get<double>("params.hyst_thresh", hyst_thresh_);

  baseline_freq_thresh_ = pt.get<double>("params.baseline_freq_thresh", baseline_freq_thresh_);
  filter_low_freq_ = pt.get<double>("params.", baseline_freq_thresh_);
  filter_high_freq_ = pt.get<double>("params.filter_high_freq",filter_high_freq_ );
  snr_thresh_ = pt.get<double>("params.snr_thresh",snr_thresh_ );
  len_thresh_ = pt.get<double>("params.len_thresh",len_thresh_ );
}

// Save the interanl TGraph.
void UserFid::SaveGraph(std::string filename, std::string title)
{
  // Grab canvas focus and clear old images.
  static TCanvas c1("c1", "FID Canvas");
  c1.cd();
  c1.Clear();

  // Draw the graph and save as a file.
  gr_.SetTitle(title.c_str());
  gr_.Draw();
  c1.Print(filename.c_str());
}


// Save a plot of FID waveform.
void UserFid::SavePlot(std::string filename, std::string title)
{
  // If no title supplied give a reasonable default.
  if (title == "") {

    title = std::string("FID; time [ms]; amplitude [a.u.]");

  } else {

    // In case they didn't append x/y labels.
    title.append("; time [ms]; amplitude [a.u.]");
  }

  gr_ = TGraph(wf_.size(), &tm_[0], &wf_[0]);

  SaveGraph(filename, title);
}


// Print the time series fit from an FID.
void UserFid::SaveTimeFit(std::string filename, std::string title)
{
  if (title == "") {

    title = std::string("Time Series Fit; time [ms]; amplitude [a.u.]");

  } else {

    // In case they didn't append x/y labels.
    title.append("; time [ms]; amplitude [a.u.]");
  }

  // Copy the current time fit graph.
  gr_ = gr_time_series_;
  SaveGraph(filename, title);
}

// Print the time series fit from an FID.
void UserFid::SaveFreqFit(std::string filename, std::string title)
{
  if (title == "") {

    title = std::string("Frequency Series Fit; time [ms]; amplitude [a.u.]");

  } else {

    // In case they didn't append x/y labels.
    title.append("; freq [kHz]; amplitude [a.u.]");
  }

  // Copy the current time fit graph.
  gr_ = gr_freq_series_;
  SaveGraph(filename, title);
}

void UserFid::SaveTimeRes(std::string filename, std::string title)
{
  if (title == "") {

    title = std::string("Time Series Fit Residuals; time [ms]; amplitude [a.u.]");

  } else {

    // In case they didn't append x/y labels.
    title.append("; time [ms]; amplitude [a.u.]");
  }

  // Copy the current time fit.
  gr_ = TGraph(res_.size());

  // Set the points
  for (uint i = 0; i < res_.size(); ++i){
    static double x, y;

    gr_time_series_.GetPoint(i, x, y);
    gr_.SetPoint(i, x, res_[i]);
  }

  SaveGraph(filename, title);
}


void UserFid::SaveFreqRes(std::string filename, std::string title)
{
  if (title == "") {

    title = std::string("Freq Series Fit Residuals; time [ms]; amplitude [a.u.]");

  } else {

    // In case they didn't append x/y labels.
    title.append("; freq [kHz]; amplitude [a.u.]");
  }

  // Copy the current time fit.
  gr_ = TGraph(res_.size());

  // Set the points
  for (uint i = 0; i < res_.size(); ++i){
    static double x, y;

    gr_freq_series_.GetPoint(i, x, y);
    gr_.SetPoint(i, x, res_[i]);
  }

  SaveGraph(filename, title);
}


// Save the FID data to a text file as "<time> <amp>".
void UserFid::SaveData(std::string filename)
{
  // open the file first
  std::ofstream out(filename);

  for (unsigned int i = 0; i < tm_.size(); ++i) {
    out << tm_[i] << " " << wf_[i] << std::endl;
  }
}


void UserFid::DiagnosticInfo(std::ostream& out)
{
  using std::endl;

  // Save the flags, set them to defaults.
  auto flags = out.flags();
  std::ofstream testout;
  out.flags(testout.flags());

  out << std::string(80, '<') << endl << std::string(4, ' ');
  out << "Diagostic Information for Fid @ " << this << endl;
  out << std::string(80, '<') << endl;

  out << "    Fid Waveform Characteristics" << endl;
  out << "        mean:       " << mean_ << endl;
  out << "        amplitude:  " << max_amp_ << endl;
  out << "        noise:      " << noise_ << endl;
  out << "        start time: " << i_wf_;
  out << " (" << tm_[i_wf_] << " ms)" << endl;
  out << "        stop time:  " << f_wf_ - 1;
  out << " (" << tm_[f_wf_ - 1] << " ms)" << endl;
  out << "        health:     " << health_ << endl;
  out << std::string(80, '>') << endl << endl;

  // Restore set flags.
  out.flags(flags);
}


void UserFid::DiagnosticPlot(std::string dirname, std::string filestub)
{
  boost::filesystem::path dir(dirname);
  boost::filesystem::create_directories(dir);

  SaveFreqFit(dir.string() + filestub + std::string("_freq_fit.png"));
  SaveTimeFit(dir.string() + filestub + std::string("_time_fit.png"));
  SaveFreqRes(dir.string() + filestub + std::string("_freq_res.png"));
  SaveTimeRes(dir.string() + filestub + std::string("_time_res.png"));
}


void UserFid::DiagnosticDump(std::string dirname, std::string filestub)
{
  // Make the plots first, that will create the directory if needed.
  DiagnosticPlot(dirname, filestub);

  std::ofstream out;
  boost::filesystem::path dir(dirname);

  std::string str = dir.string() + std::string("libfid.log");
  out.open(str , std::ofstream::out | std::ofstream::app);

  DiagnosticInfo(out);
  out.close();
}

} // fid
