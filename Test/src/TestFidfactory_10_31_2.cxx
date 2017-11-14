#include <iostream>
#include <TGraph.h>
#include <TF1.h>
#include <TFile.h>
#include <chrono>
#include "TCanvas.h"
#include "TAxis.h"
#include <TGraph2D.h>

#include "fid.h"
#include "fidfactory.h"
#include "dsp.h"
using namespace std;

int main(){
  //Fidfactory
  fid::FidFactory myfidfactory;

  vector<double> tm;
  vector<double> wf;
  vector<double> fq[10][10];

  for (int i=0;i<8.0e4;i++){
    tm.push_back(1e-7*i);
  }

  //fq.push_back(ave_fr);

  vector<double> a; // a represents curvature
  for (int i=0;i<10;i++){
    a.push_back(-0.4+0.8*i/10);
  }
  vector<double> b; // b represents slope
  for (int i=0;i<10;i++){
    b.push_back(-15+30*i/10);
  }

  //double PD_freq[100] = {0};
  //double ZC_freq[100] = {0};

  TGraph2D *gOnlyCurv_PD = new TGraph2D();
  gOnlyCurv_PD->SetName("OnlyCurvaturePD");
  gOnlyCurv_PD->SetTitle("Deviation Frequency using PD method vs Curvature & Slope");

  TGraph2D *gOnlyCurv_ZC = new TGraph2D();
  gOnlyCurv_ZC->SetName("OnlyCurvatureZC");
  gOnlyCurv_ZC->SetTitle("Deviation Frequency using ZC method vs Curvature & Slope");

  int ii = 0;

  for (int j=0;j<10;j++){
  for (int s=0;s<10;s++){
    double ave_fr = 0;
    for (int k=0;k<101;k++){
      double fr = 47000 + a[j]*(-2.5+0.05*k)*(-2.5+0.05*k)+b[s]*(-2.5+0.05*k);
      fq[j][s].push_back(fr);
      ave_fr += fr;
    }
    ave_fr /= 101;

    myfidfactory.SetGamma1(1000./3);
    myfidfactory.SetSignalToNoise(100);
    myfidfactory.SetRfDuration(0);
    //myfidfactory.SetFidFreq(50000);
    myfidfactory.SetBaseline(0);
    myfidfactory.SetAddNoise(false);
  
    myfidfactory.GradientFid_Bugao(wf,tm,fq[j][s]);

    fid::Fid myFid(wf,tm);
    //myFid.SetParameter("baseline_freq_thresh",f[j]);
    //myFid.SetParameter("filter_low_freq",f[j]);
    //myFid.SetParameter("filter_high_freq",5.0e7);
    myFid.Init("Standard");

    double PD_freq = myFid.GetFreq("PD") - ave_fr;
    double ZC_freq = myFid.GetFreq("ZC") - ave_fr;

    gOnlyCurv_PD->SetPoint(ii,a[j],b[s],PD_freq);
    gOnlyCurv_ZC->SetPoint(ii,a[j],b[s],ZC_freq);

    ii += 1;
    //cout<<PD_freq[i]<<endl;
    //cout<<ZC_freq[i]<<endl;

    //c1->Write();

    //fout->Close();
  }
  }
  
  TFile * fout = new TFile("testout_10_31_2.root","recreate");

  gOnlyCurv_PD->Write();
  gOnlyCurv_ZC->Write();

  TCanvas *c = new TCanvas("c","Graph2D example",0,0,700,600);
  c->Divide(2,1);
  c->cd(1);
  gOnlyCurv_PD->Draw("tri1 p0");
  c->cd(2);
  gOnlyCurv_ZC->Draw("tri1 p0");

  c->Write();
  fout->Close();
}
  
  
