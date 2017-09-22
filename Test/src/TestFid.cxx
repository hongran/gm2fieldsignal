#include <iostream>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>

#include "fid.h"

int main(){
  TFile * FileIn = new TFile("data/TrolleyGraphOut01023.root","read");
  TDirectory * Dir = (TDirectory*)FileIn->Get("FidPulsePlot");
  TGraph * gWf = (TGraph*)Dir->Get("Pulse_3158");
  gWf->SetName("Waveform");

  //Get Arrays
  auto X = gWf->GetX();
  auto Y = gWf->GetY();
  auto N = gWf->GetN();

  std::vector<double> V;
  std::vector<double> T;

  //output text
  std::ofstream fileout;
  fileout.open("trolley_fid.txt",std::ios::out);

  for (int i=0;i<N;i++){
    fileout<<X[i]<<" "<<Y[i]<<std::endl;
  }
  fileout.close();

  for (int i=500;i<N-800;i++){
    V.push_back(Y[i]);
    T.push_back(X[i]);
  }

  //Fid
  fid::Fid myFid(V,T);
  myFid.SetParameter("baseline_freq_thresh",1);
  myFid.SetParameter("start_amplitude",0.3);
  myFid.SetParameter("filter_low_freq",20000);
  myFid.SetParameter("filter_high_freq",90000);
  myFid.SetParameter("fft_peak_width",40000);
  for (int i=0;i<10;i++){
    fid::Fid myFid3(V,T);
    auto t0 = std::chrono::high_resolution_clock::now();
    myFid3.Init("Standard");
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
    double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
    std::cout << "Time = "<<t<<std::endl;
  }
  myFid.Init("Standard");
  std::cout << myFid.GetFreq("PD")<<std::endl;
  std::cout << myFid.GetFreq("LZ")<<std::endl;
  std::cout << myFid.amp()<<std::endl;
  std::cout << myFid.snr()<<std::endl;
  myFid.CalcLorentzianFreq();
  auto FitFunc = myFid.f_fit();
  std::cout << myFid.i_wf()<<std::endl;
  std::cout << myFid.f_wf()<<std::endl;

  //auto t0 = std::chrono::high_resolution_clock::now();
  std::cout << myFid.GetFreq("ZC")<<std::endl;
 /* auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
  double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
  std::cout << "ZC Time = "<<t<<std::endl;
  */

  fid::Fid myFid2(V,T);
  myFid2.SetParameter("baseline_freq_thresh",200);
  myFid2.SetParameter("filter_low_freq",30000);
  myFid2.SetParameter("filter_high_freq",70000);
  myFid2.Init("Old");
  //std::cout << myFid2.GetFreq("PH")<<std::endl;
  //std::cout << myFid2.GetFreq("LZ")<<std::endl;


  auto Psd = myFid.psd();
  auto Baseline = myFid.baseline();
  auto Wf_filter = myFid.filtered_wf();
  auto Wf_im = myFid.wf_im();
  auto Phi = myFid.phi();
  auto Env = myFid.env();
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


  FileIn->Close();
  delete FileIn;
  TFile * FileOut = new TFile("TestOut.root","recreate");
  gWf->Write();
  gPsd->Write();
  gBaseline->Write();
  gWf_filter->Write();
  gWf_im->Write();
  gPhi->Write();
  gEnv->Write();
  gEnvN->Write();
  FitFunc.Write();
  FileOut->Close();
  delete FileOut;

  return 0;
}
