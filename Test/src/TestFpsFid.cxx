#include <iostream>
#include <fstream>
#include <TROOT.h>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>

#include "fid.h"

int main(){
  TGraph * gWf = new TGraph();

  int run  = 1280;

  TFile* filein = new TFile(Form("data/FpsFidGraphOut%05d_tier0.root",run),"read");
  TDirectory * d1 = (TDirectory*)filein->Get("PlotFixedProbeFid");
  gROOT->cd();
  int EventID = 2;
  int ProbeID = 150;
  TGraph *g = (TGraph*)d1->Get(Form("Event_%03d_probe_%03d",EventID,ProbeID));
  gWf = (TGraph*)g->Clone();

  filein->Close();
  delete filein;

  gWf->SetName("Waveform");
  //Get Arrays
  auto X = gWf->GetX();
  auto Y = gWf->GetY();
  auto N = gWf->GetN();

  std::vector<double> V;
  std::vector<double> T;

  for (int i=0;i<N-800;i++){
    V.push_back(Y[i]);
    T.push_back(X[i]);
  }

  //Fid
  fid::Fid myFid(V,T);
  myFid.SetParameter("baseline_freq_thresh",500);
  myFid.SetParameter("start_amplitude",0.37);
  myFid.SetParameter("filter_low_freq",500);
  myFid.SetParameter("filter_high_freq",5000000);
  myFid.SetParameter("fft_peak_width",40000);
  myFid.SetParameter("edge_width",2e-5);
  myFid.SetParameter("edge_ignore",6e-5);
  myFid.SetParameter("hyst_thresh",0.7);
  for (int i=0;i<10;i++){
    fid::Fid myFid3(V,T);
    auto t0 = std::chrono::high_resolution_clock::now();
  //  myFid3.Init("FFTOnly");
    myFid3.Init("Standard");
  //  myFid3.BaselineCorrection();
  //  myFid3.FindFidRange();
  //  myFid3.GuessFitParams();
  //  myFid3.CalcSnrFFT();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
    double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
    std::cout << "Time = "<<t<<std::endl;
  }
  myFid.Init("Standard");
  std::cout<<"PD: " << myFid.GetFreq("PD")<<std::endl;
  std::cout<<"ZC: " << myFid.GetFreq("ZC")<<std::endl;
//  std::cout << myFid.GetFreq("LZ")<<std::endl;
  std::cout << myFid.amp()<<std::endl;
  std::cout << myFid.snr()<<std::endl;
//  myFid.CalcLorentzianFreq();
  auto FitFunc = myFid.f_fit();
  std::cout << myFid.i_wf()<<std::endl;
  std::cout << myFid.f_wf()<<std::endl;

  auto Psd = myFid.psd();
  auto Baseline = myFid.baseline();
  auto Wf_filter = myFid.filtered_wf();
  auto Wf_corrected = myFid.wf();
  auto Wf_im = myFid.wf_im();
  auto Phi = myFid.phi();
  auto Env = myFid.env();
  auto Res = myFid.res();
  auto freq = myFid.fftfreq();
  auto tm = myFid.tm();

  TGraph * gPsd = new TGraph();
  gPsd->SetName("Psd");

  for (int i=0;i<Psd.size();i++){
    gPsd->SetPoint(i,freq[i],Psd[i]);
  }

  TGraph * gBaseline = new TGraph();
  gBaseline->SetName("Baseline");

  for (int i=0;i<Baseline.size();i++){
    gBaseline->SetPoint(i,tm[i],Baseline[i]);
  }

  TGraph * gWf_filter = new TGraph();
  gWf_filter->SetName("Wf_filter");

  for (int i=0;i<Wf_filter.size();i++){
    gWf_filter->SetPoint(i,tm[i],Wf_filter[i]);
  }

  TGraph * gWf_corrected = new TGraph();
  gWf_corrected->SetName("Wf_corrected");

  for (int i=0;i<Wf_corrected.size();i++){
    gWf_corrected->SetPoint(i,tm[i],Wf_corrected[i]);
  }

  TGraph * gWf_im = new TGraph();
  gWf_im->SetName("Wf_im");

  for (int i=0;i<Wf_im.size();i++){
    gWf_im->SetPoint(i,tm[i],Wf_im[i]);
  }

  TGraph * gPhi = new TGraph();
  gPhi->SetName("Phi");

  for (int i=0;i<Phi.size();i++){
    gPhi->SetPoint(i,tm[i],Phi[i]);
  }

  TGraph * gEnv = new TGraph();
  gEnv->SetName("Env");
  TGraph * gEnvN = new TGraph();
  gEnvN->SetName("EnvN");

  for (int i=0;i<Env.size();i++){
    gEnv->SetPoint(i,tm[i],Env[i]);
    gEnvN->SetPoint(i,tm[i],-Env[i]);
  }

  TGraph * gRes = new TGraph();
  gRes->SetName("gRes");

  for (int i=0;i<Res.size();i++){
    gRes->SetPoint(i,tm[i],Res[i]);
  }


  TFile * FileOut = new TFile("TestOut.root","recreate");
  gWf->Write();
  gPsd->Write();
  gBaseline->Write();
  gWf_corrected->Write();
  gWf_filter->Write();
  gWf_im->Write();
  gPhi->Write();
  gEnv->Write();
  gEnvN->Write();
  gRes->Write();
  FitFunc.Write();
  FileOut->Close();
  delete FileOut;

  return 0;
}
