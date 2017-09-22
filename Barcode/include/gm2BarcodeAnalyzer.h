/*
 * =====================================================================================
 *       Filename:  gm2BarcodeAnalyzer.h
 *
 *    Description:  Barcode classes for the GM2 experiment
 *
 *        Version:  1.0
 *        Created:  03/24/2016 10:21:45
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ran Hong 
 *   Organization:  ANL
 
 * =====================================================================================
*/
#include <map>
#include <memory>
#include <string>

using namespace std;

namespace gm2field{
  class LogicLevel{
    public:
      int Level;
      size_t LEdge;	//Index of left edge
      size_t REdge;	//Index of right edge
  };
  class AbsBarcodeSegment{
    public:
      int RegionType;		//Code region (1) or non-code region(0) or abnormal region(-1)
      int Code;			//Code represented by the binary number
      size_t NAbsLevel;		//Number of Abs levels
      size_t NRegLevel;		//Number of Reg levels
      vector<size_t> fLevelIndexList;	//List of Abs logic level indicies
      vector<LogicLevel> fRegLevelList;	//List of Reg logic levels
  };

  enum AbnormalStatus{
    None,
    ZeroVelocity,
    StartRegion,
    EndRegion,
    ChamberJunction,
    BigGap
  };

  class TBarcode{
    public:
      vector<double>* fXTime;
      vector<double>* fXPos;
      vector<float>* fY;
      size_t fNPoints;		//Number of points
      string fType;

      vector<size_t> fMaxList;	//List of maximum time stamps
      vector<size_t> fMinList;	//List of minimum time stamps
      vector<size_t> fExtremaList;	//List of extremum time stamps
      vector<gm2field::AbnormalStatus> fAbnormal;
      vector<gm2field::LogicLevel> fLogicLevels;	//logic levels converted from the analog signal
      size_t fNHighLevels;		//Number of high levels
      size_t fNLowLevels;		//Number of low levels
      size_t fNExtrema;		//Number o extrema

      bool LogicLevelConverted;	//Whether logic levels are converted
      bool ExtremaFound;		//Whether Extrema are found
      
      //Only for Reg
      vector<float> fAverage;
      vector<float> fContrast;
      vector<size_t> fBigGapList;

      //Only for Abs
      vector<vector<gm2field::LogicLevel>> fAuxList;	//Auxilliary list of RegLevels associated with each AbsLevel
      vector<gm2field::AbsBarcodeSegment> fSegmentList;
      size_t fNSegments;
      bool fSegmented;
  };

  class gm2BarcodeAnalyzer{
    private:
      //Attributes
      //Parameters
      float fThreshold;          //Threshold
      float fTransitionThreshold;//Threshold for determining the transition between high and low level
      float fLogicLevelScale;    //Value of logic-1 for plotting graph

      //Containers
      map<string,TBarcode> BarcodeList;
      //Private methods
      int FindHalfRise(string name, int low, int high);	//Find half-way rising edge
      int FindHalfFall(string name, int high, int low);	//Find half-way falling edge

    public:
      gm2BarcodeAnalyzer();
      ~gm2BarcodeAnalyzer();
      //Set Methods
      int RegisterBarcode(string name, string Type, vector<float>* fY, vector<double>* fXTime, vector<double>* fXPos);
      void SetThreshold(const float val){fThreshold = val;}
      void SetTransitionThreshold(const float val);
      void SetLogicLevelScale(const float Level){fLogicLevelScale = Level;}
      //Get Methods
      float GetThreshold() const{return fThreshold;}
      bool IfLogicLevelConverted(string name) {return BarcodeList[name].LogicLevelConverted;}
      bool IfExtremaFound(string name) {return BarcodeList[name].ExtremaFound;}
      size_t GetNPoints(string name) {return BarcodeList[name].fNPoints;}
      size_t GetNHighLevels(string name) {return BarcodeList[name].fNHighLevels;}
      size_t GetNLowLevels(string name) {return BarcodeList[name].fNLowLevels;}
      size_t GetNExtrema(string name) {return BarcodeList[name].fNExtrema;}
      vector<size_t> GetExtremaList(string name) {return BarcodeList[name].fExtremaList;}
      vector<gm2field::LogicLevel> GetLogicLevels(string name) {return BarcodeList[name].fLogicLevels;}
      //for Reg
      vector<float> GetAverage(string name) {return BarcodeList[name].fAverage;}
      vector<float> GetContrast(string name) {return BarcodeList[name].fContrast;}
      //Reset
      int Reset();

      //Modifiers
      void Smooth(string name,size_t PointsAveraged = 5);

      //Analysis
      int FindExtrema(string name);
      int ConvertToLogic(string name);
      int AnalyzeAll();
      //for Reg
      int FindBigGaps(string name);
      //for Abs
      int ChopSegments(string name,string reference);
      int Decode(string name);
  };

}
