#include <iostream>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>
#include "TCanvas.h"

#include "fid.h"
#include "fidfactory.h"
#include "dsp.h"
using namespace std;

int main(){
  //Fidfactory
  fid::FidFactory myfidfactory;

  vector<double> tm;
  vector<double> wf;

  for (int i=0;i<4.0e4;i++){
  tm.push_back(1e-7*i);
  }

  myfidfactory.SetGamma1(1000./3);
  myfidfactory.SetSignalToNoise(100);
  myfidfactory.SetRfDuration(0);
  myfidfactory.SetFidFreq(50000);
  myfidfactory.SetBaseline(0);
  myfidfactory.SetAddNoise(false);

  TGraph * gIdealFid = new TGraph();
  gIdealFid->SetName("IdealFidSignal");
  gIdealFid->SetTitle("Ideal Fid Signal");
  
  myfidfactory.IdealFid(wf,tm);
  for (int i=0;i<tm.size();i++){
    gIdealFid->SetPoint(i,tm[i],wf[i]);
  }

  TFile * fout = new TFile("testout.root","recreate");
  gIdealFid->Write();


  //Fid
  fid::Fid myFid(wf,tm);

  myFid.SetParameter("baseline_freq_thresh",10000);
  myFid.SetParameter("filter_low_freq",10000);
  myFid.SetParameter("filter_high_freq",5.0e7);
  myFid.Init("Standard");

  auto tm2 = tm;
  auto wf2 = wf;

  fid::Fid myFid2(wf2,tm2);
  myFid2.Init("Speed");

  std::cout << myFid.GetFreq("PD")<<std::endl;
  std::cout << myFid.GetFreq("LZ")<<std::endl;
  std::cout << myFid2.GetFreq("ZC")<<std::endl;
  std::cout << myFid.GetFreq("ZC")<<std::endl;
  
  auto Baseline = myFid.baseline();
  auto Phi = myFid.phi();

  TGraph * gPhi = new TGraph();
  gPhi->SetTitle("Phi");
  gPhi->SetName("gPhi");

  for (int i=0;i<Phi.size();i++){
    gPhi->SetPoint(i,tm[i],Phi[i]);
  }
  gPhi->Write();
  
  TGraph * gBaseline = new TGraph();
  gBaseline->SetTitle("Baseline calculated by Fit");
  gBaseline->SetName("BaselinecalFit");

  for (int i=0;i<Baseline.size();i++){
    gBaseline->SetPoint(i,tm[i],Baseline[i]);
  }
  gBaseline->Write();

  TGraph * gBaseline_2 = new TGraph();
  gBaseline_2->SetTitle("Baseline Filtered");
  gBaseline_2->SetName("BaselineFiltered");

  TGraph * gFiltered = new TGraph();
  gFiltered->SetTitle("Filtered Wf");
  gFiltered->SetName("gFiltered");

  TGraph * gCorrected = new TGraph();
  gCorrected->SetTitle("Corrected Wf");
  gCorrected->SetName("gCorrected");

  auto Wf_filter = myFid.filtered_wf();
  auto Wf_corrected = myFid.wf();

  for (int i=0;i<Wf_filter.size();i++){
    gBaseline_2->SetPoint(i,tm[i],wf[i]-Wf_filter[i]);
    gFiltered->SetPoint(i,tm[i],Wf_filter[i]);
  }

  gFiltered->Write();

  for (int i=0;i<Wf_corrected.size();i++){
    gCorrected->SetPoint(i,tm[i],Wf_corrected[i]);
  }
  gCorrected->Write();

  gBaseline_2->Fit("pol0");

  gBaseline_2->Write();

  fout->Close();

}
  
  
