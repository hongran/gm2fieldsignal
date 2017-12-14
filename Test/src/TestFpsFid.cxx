#include <iostream>
#include <fstream>
#include <TROOT.h>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>

#include "fid.h"
#include <stdlib.h>

int main(int argc,char ** argv){
  TGraph * gWf = new TGraph();

  int run  = atoi(argv[1]);
  int EventID = atoi(argv[2]);
  int ProbeStart = atoi(argv[3]);
  unsigned int NBatch = atoi(argv[4]);
  unsigned int fid_size = 40960;

  std::vector<double> V;
  std::vector<double> T;

  TFile* filein = new TFile(Form("/home/newg2/DataProduction/Nearline/ArtTFSDir/FpsFidGraphOut%05d_tier0.root",run),"read");
  TDirectory * d1 = (TDirectory*)filein->Get("PlotFixedProbeFullFid");
  gROOT->cd();
 
  for (unsigned int ProbeID=ProbeStart;ProbeID<ProbeStart+NBatch;ProbeID++){
    TGraph *g = (TGraph*)d1->Get(Form("Event_%03d_probe_%03d",EventID,ProbeID));
    //Get Arrays
    auto X = g->GetX();
    auto Y = g->GetY();
    auto N = g->GetN();

    std::cout << "Probe "<<ProbeID<<" Size "<<N<<std::endl;
    for (unsigned int i=0;i<fid_size;i++){
      V.push_back(Y[i]);
      T.push_back(X[i]);
    }
  }

  filein->Close();
  delete filein;

  std::cout << V.size()<<std::endl;
  for (unsigned int j=0;j<V.size();j++){
    gWf->SetPoint(j,T[j]+(j/fid_size)*(T[fid_size-1]-T[0]),V[j]);
  }
  gWf->SetName("Waveform");

  //Fid
  
  fid::Fid myFid(V,T[1]-T[0],NBatch,fid_size);
  myFid.SetParameter("baseline_freq_thresh",500);
  myFid.SetParameter("start_amplitude",0.37);
  myFid.SetParameter("filter_low_freq",500);
  myFid.SetParameter("filter_high_freq",5000000);
  myFid.SetParameter("fft_peak_width",40000);
  myFid.SetParameter("edge_width",2e-5);
  myFid.SetParameter("edge_ignore",6e-5);
  myFid.SetParameter("hyst_thresh",0.7);
  
  /*
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
  }*/
  auto t0 = std::chrono::high_resolution_clock::now();
  myFid.Init("Standard");
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dtn = t1.time_since_epoch() - t0.time_since_epoch();
  double t = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn).count();
  std::cout << "Time = "<<t<<std::endl;
  myFid.Init("Standard");
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dtn2 = t2.time_since_epoch() - t1.time_since_epoch();
  double dt2 = std::chrono::duration_cast<std::chrono::nanoseconds>(dtn2).count();
  std::cout << "Time = "<<dt2<<std::endl;
/*  std::cout<<"PD: " << myFid.GetFreq("PD")<<std::endl;
  std::cout<<"ZC: " << myFid.GetFreq("ZC")<<std::endl;
  auto t0lz = std::chrono::high_resolution_clock::now();
  std::cout<<"LZ: " << myFid.GetFreq("LZ")<<std::endl;
  auto t1lz = std::chrono::high_resolution_clock::now();
  auto dtnlz = t1lz.time_since_epoch() - t0lz.time_since_epoch();
  double tlz = std::chrono::duration_cast<std::chrono::nanoseconds>(dtnlz).count();
  std::cout << "Time LZ = "<<tlz<<std::endl;
  std::cout << myFid.amp()<<std::endl;
  std::cout << myFid.snr()<<std::endl;
//  myFid.CalcLorentzianFreq();
  std::cout << myFid.i_wf()<<std::endl;
  std::cout << myFid.f_wf()<<std::endl;
*/
  
  auto Psd = myFid.psd();
  auto Baseline = myFid.baseline();
  auto Wf_filter = myFid.filtered_wf();
  auto Wf_corrected = myFid.wf();
  auto Wf_im = myFid.wf_im();
  auto Phi = myFid.phi();
  auto Env = myFid.env();
//  auto Res = myFid.res();
  auto freq = myFid.fftfreq();
  auto tm = myFid.tm();

  TGraph * gPsd = new TGraph();
  gPsd->SetName("Psd");

  for (unsigned int i=0;i<Psd.size();i++){
    //gPsd->SetPoint(i,freq[i]+(i/fid_size)*(freq[fid_size-1]-freq[0]),Psd[i]);
    gPsd->SetPoint(i,i,Psd[i]);
  }

  TGraph * gBaseline = new TGraph();
  gBaseline->SetName("Baseline");

  for (unsigned int i=0;i<Baseline.size();i++){
    gBaseline->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Baseline[i]);
  }

  TGraph * gWf_filter = new TGraph();
  gWf_filter->SetName("Wf_filter");

  for (unsigned int i=0;i<Wf_filter.size();i++){
    gWf_filter->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Wf_filter[i]);
  }

  TGraph * gWf_corrected = new TGraph();
  gWf_corrected->SetName("Wf_corrected");

  for (unsigned int i=0;i<Wf_corrected.size();i++){
    gWf_corrected->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Wf_corrected[i]);
  }

  TGraph * gWf_im = new TGraph();
  gWf_im->SetName("Wf_im");

  for (unsigned int i=0;i<Wf_im.size();i++){
    gWf_im->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Wf_im[i]);
  }

  TGraph * gPhi = new TGraph();
  gPhi->SetName("Phi");

  for (unsigned int i=0;i<Phi.size();i++){
    gPhi->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Phi[i]);
  }

  TGraph * gEnv = new TGraph();
  gEnv->SetName("Env");
  TGraph * gEnvN = new TGraph();
  gEnvN->SetName("EnvN");

  for (unsigned int i=0;i<Env.size();i++){
    gEnv->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),Env[i]);
    gEnvN->SetPoint(i,tm[i]+(i/fid_size)*(tm[fid_size-1]-tm[0]),-Env[i]);
  }
/*
  TGraph * gRes = new TGraph();
  gRes->SetName("gRes");

  for (int i=0;i<Res.size();i++){
    gRes->SetPoint(i,tm[i],Res[i]);
  }
*/

  TGraph * gPhiFit[400];
  TF1 FitFunc[400];
  for (unsigned int j=0;j<NBatch;j++){
    gPhiFit[j] = new TGraph();
    gPhiFit[j]->SetName(Form("PhiFit%d",j));
    for (unsigned int i=0;i<fid_size;i++){
      gPhiFit[j]->SetPoint(i,tm[j*fid_size+i],Phi[j*fid_size+i]);
    }
    FitFunc[j] = myFid.f_fit(j);
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
 // gRes->Write();
  for (unsigned int j=0;j<NBatch;j++){
    gPhiFit[j]->Write();
    FitFunc[j].Write();
  }
  FileOut->Close();
  delete FileOut;

  for (unsigned int j=0;j<NBatch;j++){
    delete gPhiFit[j];
  }
  return 0;
}
