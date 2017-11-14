#include <iostream>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>
#include "TCanvas.h"
#include "TAxis.h"

#include "fid.h"
#include "fidfactory.h"
#include "dsp.h"
using namespace std;

int main(){
  //Fidfactory
  fid::FidFactory myfidfactory;

  vector<double> tm;
  vector<double> wf;
  vector<double> fq[100];

  for (int i=0;i<8.0e4;i++){
    tm.push_back(1e-7*i);
  }

  //fq.push_back(ave_fr);

  vector<double> a;
  for (int i=0;i<10;i++){
    a.push_back(-0.4+0.8*i/10);
  }
  vector<double> b;
  for (int i=0;i<10;i++){
    b.push_back(-15+30*i/10);
  }

  //double PD_freq[100] = {0};
  //double ZC_freq[100] = {0};

  TGraph * gOnlyCurv_PD = new TGraph();
  gOnlyCurv_PD->SetName("OnlyCurvaturePD");
  gOnlyCurv_PD->SetTitle("Deviation Frequency using PD method vs Curvature");

  TGraph * gOnlyCurv_ZC = new TGraph();
  gOnlyCurv_ZC->SetName("OnlyCurvatureZC");
  gOnlyCurv_ZC->SetTitle("Deviation Frequency using ZC method vs Curvature");

  for (int j=0;j<10;j++){
    double ave_fr = 0;
    for (int k=0;k<101;k++){
      double fr = 47000 + a[j]*(-2.5+0.05*k)*(-2.5+0.05*k)+0*(-2.5+0.05*k);// no slope
      fq[j].push_back(fr);
      ave_fr += fr;
    }
    ave_fr /= 101;

    myfidfactory.SetGamma1(1000./3);
    myfidfactory.SetSignalToNoise(100);
    myfidfactory.SetRfDuration(0);
    //myfidfactory.SetFidFreq(50000);
    myfidfactory.SetBaseline(0);
    myfidfactory.SetAddNoise(false);
  
    myfidfactory.GradientFid_Bugao(wf,tm,fq[j]);

    fid::Fid myFid(wf,tm);
    //myFid.SetParameter("baseline_freq_thresh",f[j]);
    //myFid.SetParameter("filter_low_freq",f[j]);
    //myFid.SetParameter("filter_high_freq",5.0e7);
    myFid.Init("Standard");

    double PD_freq = myFid.GetFreq("PD") - ave_fr;
    double ZC_freq = myFid.GetFreq("ZC") - ave_fr;

    gOnlyCurv_PD->SetPoint(j,a[j],PD_freq);
    gOnlyCurv_ZC->SetPoint(j,a[j],ZC_freq);
    //cout<<PD_freq[i]<<endl;
    //cout<<ZC_freq[i]<<endl;
  }
  
  TFile * fout = new TFile("testout_10_31.root","recreate");
  gOnlyCurv_PD->Write();
  gOnlyCurv_ZC->Write();
  fout->Close();
}
  
  
