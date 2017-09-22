#ifndef USERFID_H
#define USERFID_H

/*============================================================================*\

author: Matthias W. Smith , Ran Hong
email: mwmsith2@uw.edu, rhong@anl.gov

notes:

  Defines a user FID class for more user utilities.

\*===========================================================================*/

//--- std includes ----------------------------------------------------------//
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

//--- other includes --------------------------------------------------------//
#include "TCanvas.h"
#include "TGraph.h"
#include "TF1.h"

#include "fid.h"

namespace fid {

// A struct useful for saving analysis results
struct fid_freq_t {
  Double_t freq[7];
  Double_t ferr[7];
  Double_t chi2[7];
  Double_t snr;
  Double_t len;
  UShort_t health;
};

const char * const fid_str = "wf[10000]/D:tm[10000]/D";

const char * const fid_freq_str =
"freq[7]/D:ferr[7]/D:chi2[7]/D:snr/D:len/D:health/s";

static std::string logdir("/var/log/fid/");

class UserFid : public Fid {

 public:

  // Default ctors/dtors.
 UserFid() : Fid() {};

  ~UserFid() {};

  // Other ctors
 UserFid(const std::vector<double>& wf, const std::vector<double>& tm) :
  Fid(wf, tm) {};

 UserFid(const std::vector<double>& wf) :
  Fid(wf) {};

  // diagnostic function
  void DiagnosticInfo(std::ostream& out=std::cout);
  void DiagnosticPlot(std::string dirname=logdir, 
                      std::string filestub="fid_diagnostics");
  void DiagnosticDump(std::string dirname=logdir, 
                      std::string filestub="fid_diagnostics");

  // Utility functions
  void SaveData(std::string filename);
  void SaveGraph(std::string filename, std::string title);
  void SavePlot(std::string filename, std::string title="");
  void SaveTimeFit(std::string filename, std::string title="");
  void SaveFreqFit(std::string filename, std::string title="");
  void SaveTimeRes(std::string filename, std::string title="");
  void SaveFreqRes(std::string filename, std::string title="");

  // internal utility functions
  void LoadTextData(std::string filename);
  void LoadParams(std::string conf_file);

 protected:
  
  // For general plotting.
  TCanvas c1_;
  TGraph gr_;
  
}; // UserFid
 
} // fid

#endif
