/*
 * =====================================================================================
 *
 *       Filename:  gm2BarcodeAnalyzer.cxx
 *
 *    Description:  gm2BarcodeAnalyzer classes function definitions.
 *
 *        Version:  1.0
 *        Created:  03/24/2016 10:21:45
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Ran Hong 
 *   Organization:  ANL
 *
 * =====================================================================================
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include "gm2BarcodeAnalyzer.h"

using namespace std;
using namespace gm2field;

/**********************************************************************/
//Base class TBarcode
/**********************************************************************/
gm2BarcodeAnalyzer::gm2BarcodeAnalyzer() : 
  fThreshold (0.5),           //Threshold
  fTransitionThreshold (0.355),//Transition threshold
  fLogicLevelScale (1.0)     //By default, logic-1 is "1"
{}

/**********************************************************************/
gm2BarcodeAnalyzer::~gm2BarcodeAnalyzer()
{}

/**********************************************************************/
int gm2BarcodeAnalyzer::Reset()
{
  BarcodeList.clear();
  return 0;
}
/**********************************************************************/
int gm2BarcodeAnalyzer::RegisterBarcode(string name, string Type, vector<float>* fY, vector<double>* fXTime, vector<double>* fXPos)
{
  if (BarcodeList.find(name) != BarcodeList.end()){
    cout << "Barcode with name " << name << " is already registered." << endl;
    return -1;
  }
  if (Type.compare("Reg")==0 || Type.compare("Abs")==0){
    BarcodeList[name].fType = Type;
  }else{
    cout << "Invalid Type "<<Type<<endl;
    return -2;
  }
  if (fY==nullptr || fXTime==nullptr || fXPos==nullptr){
    cout << "Invalid Barcode Timestam, position or readout pointers." << endl;
    return -3;
  }
  BarcodeList[name].fY = fY;
  BarcodeList[name].fXTime = fXTime;
  BarcodeList[name].fXPos = fXPos;

  if (BarcodeList[name].fY->size()==BarcodeList[name].fXTime->size() && BarcodeList[name].fY->size()==BarcodeList[name].fXPos->size()){
    BarcodeList[name].fNPoints = BarcodeList[name].fY->size();
  }else{
    cout << "Barcode Timestam, position and readout vectors do not have the same size."<<endl;
    return -4;
  }

  //Other parameters
  BarcodeList[name].fNHighLevels = 0;
  BarcodeList[name].fNLowLevels = 0;
  BarcodeList[name].fNExtrema = 0; 

  BarcodeList[name].LogicLevelConverted = false;
  BarcodeList[name].ExtremaFound = false;      

  BarcodeList[name].fNSegments = 0;
  BarcodeList[name].fSegmented = false;

  return 0;
}

/**********************************************************************/
void gm2BarcodeAnalyzer::SetTransitionThreshold(const float val)
{
  if (val>0.0 && val<1.0){
    fTransitionThreshold = val;
  }else{
    cout <<"Warning! TransitionThreshold should be in region (0,1). Set back to default 0.355.";
    fTransitionThreshold = 0.355;
  }
}

/**********************************************************************/
int gm2BarcodeAnalyzer::FindHalfRise(string name, int low, int high)
{
  float Difference = BarcodeList[name].fY->at(high)-BarcodeList[name].fY->at(low);
  int i=low;
  for (i=low;i<=high;i++){
    if (BarcodeList[name].fY->at(i)-BarcodeList[name].fY->at(low)>Difference*fTransitionThreshold)break;
  }
  return i;
}

/**********************************************************************/
int gm2BarcodeAnalyzer::FindHalfFall(string name, int high, int low)
{
  float Difference = BarcodeList[name].fY->at(high)-BarcodeList[name].fY->at(low);
  int i=high;
  for (i=high;i<=low;i++){
    if (BarcodeList[name].fY->at(i)-BarcodeList[name].fY->at(low)<Difference*fTransitionThreshold)break;
  }
  return i;
}

/**********************************************************************/
//Reg Barcode
/**********************************************************************/
int gm2BarcodeAnalyzer::FindExtrema(string name)
{
  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return -1;
  }
  auto N = BarcodeList[name].fXTime->size();
  int MaxN = 0;
  int MinN = 0;
  BarcodeList[name].fNExtrema = 0;
  BarcodeList[name].fMaxList.clear();
  BarcodeList[name].fMinList.clear();

  //Read out values
  float y1,y2;

  int MaxIndex = 0;
  float LocalMax = -1E6;
  int MinIndex = 0;
  float LocalMin = 1E6;

  if (BarcodeList[name].fType.compare("Reg")==0){
    //Contrast threshold method
    //To determine a minimum or maximum, the depth of the peak of trough needs to be larger than the Threshold
    //Find initial trend
    bool LastWasMax = false;
    if(N > 4){
      y1 = BarcodeList[name].fY->at(0);
      y2 = BarcodeList[name].fY->at(4);
      if(y1 > y2){
	LastWasMax = true;
      }else{
	LastWasMax = false;
      }
    }else{
      cout <<"Barcode length is too short."<<endl;
      return -1;
    }

    for(size_t i=0; i<N-1; i++){
      y1 = BarcodeList[name].fY->at(i);
      y2 = BarcodeList[name].fY->at(i+1);

      if(y1 < LocalMin && LastWasMax){
	MinIndex = i;
	LocalMin = y1;
      }
      if(y1 > LocalMax && !LastWasMax){
	MaxIndex = i;
	LocalMax = y1;
      }

      if(y1 > LocalMin && y2 > LocalMin+fThreshold && y2 > y1 && LastWasMax){
	//	cout <<"min "<<MinIndex<<" "<<LocalMin<<endl;
	BarcodeList[name].fMinList.push_back(MinIndex);
	BarcodeList[name].fExtremaList.push_back(MinIndex);
	MinN++;
	LocalMin = 1E6;
	LastWasMax = false;
      }
      if(y1 < LocalMax && y2 < LocalMax-fThreshold && y2 < y1 && !LastWasMax){
	//	cout <<"max "<<MaxIndex<<" "<<LocalMax<<endl;
	BarcodeList[name].fMaxList.push_back(MaxIndex);
	BarcodeList[name].fExtremaList.push_back(MaxIndex);
	MaxN++;
	LocalMax = -1E6;
	LastWasMax = true;
      }
    }
    BarcodeList[name].fNExtrema = MaxN + MinN;
    BarcodeList[name].ExtremaFound = true;

    //Calculate Average and Contrast
    BarcodeList[name].fAverage.resize(BarcodeList[name].fNPoints,-1);
    BarcodeList[name].fContrast.resize(BarcodeList[name].fNPoints,-1);
    for (size_t i=0;i<BarcodeList[name].fNExtrema-1;i++){
      float average_val = (BarcodeList[name].fY->at(BarcodeList[name].fExtremaList[i])+BarcodeList[name].fY->at(BarcodeList[name].fExtremaList[i+1]))/2.0;
      if (average_val<=0){
	cerr<<"Error! Average Barcode signal is zero!"<<endl;
	return -1;
      }
      float contrast_val = abs(BarcodeList[name].fY->at(BarcodeList[name].fExtremaList[i])-BarcodeList[name].fY->at(BarcodeList[name].fExtremaList[i+1]))/average_val/2.0;
      for (size_t j=BarcodeList[name].fExtremaList[i];j<BarcodeList[name].fExtremaList[i+1];j++){
	BarcodeList[name].fAverage[j] = average_val;
	BarcodeList[name].fContrast[j] = contrast_val;
      }
    }
    //handle the beginning section and ending section
    for (size_t j=0;j<BarcodeList[name].fExtremaList[0];j++){
      BarcodeList[name].fAverage[j] = BarcodeList[name].fAverage[BarcodeList[name].fExtremaList[0]];
      BarcodeList[name].fContrast[j] = BarcodeList[name].fContrast[BarcodeList[name].fExtremaList[0]];
    }
    for (size_t j=BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-1];j<N;j++){
      BarcodeList[name].fAverage[j] = BarcodeList[name].fAverage[BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-2]];
      BarcodeList[name].fContrast[j] = BarcodeList[name].fContrast[BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-2]];
    }
  }else if(BarcodeList[name].fType.compare("Abs")==0){
    //Contrast threshold method
    //To determine a minimum or maximum, the depth of the peak of trough needs to be larger than the Threshold
    //Find initial trend
    bool LastWasMax = false;
    if(N > 4){
      y1 = BarcodeList[name].fY->at(0);
      y2 = BarcodeList[name].fY->at(4);
      if(y1 > y2){
	LastWasMax = true;
      }else{
	LastWasMax = false;
      }
    }else{
      cout <<"Barcode length is too short."<<endl;
      return -1;
    }

    for(size_t i=0; i<N-1; i++){
      y1 = BarcodeList[name].fY->at(i);
      y2 = BarcodeList[name].fY->at(i+1);

      if(y1 < LocalMin && LastWasMax){
	MinIndex = i;
	LocalMin = y1;
      }
      if(y1 > LocalMax && !LastWasMax){
	MaxIndex = i;
	LocalMax = y1;
      }

      if(y1 > LocalMin && y2 > LocalMin+fThreshold && y2 > y1 && LastWasMax){
	//	cout <<"min "<<MinIndex<<" "<<LocalMin<<endl;
	BarcodeList[name].fMinList.push_back(MinIndex);
	BarcodeList[name].fExtremaList.push_back(MinIndex);
	MinN++;
	LocalMin = 1E6;
	LastWasMax = false;
      }
      if(y1 < LocalMax && y2 < LocalMax-fThreshold && y2 < y1 && !LastWasMax){
	//	cout <<"max "<<MaxIndex<<" "<<LocalMax<<endl;
	BarcodeList[name].fMaxList.push_back(MaxIndex);
	BarcodeList[name].fExtremaList.push_back(MaxIndex);
	MaxN++;
	LocalMax = -1E6;
	LastWasMax = true;
      }
    }
    BarcodeList[name].fNExtrema = MaxN + MinN;
    BarcodeList[name].ExtremaFound = true;
  }else{
    cout << "Invalid Type " << BarcodeList[name].fType << endl;
    return -1;
  }
  return 0;
}
/**********************************************************************/
int gm2BarcodeAnalyzer::FindBigGaps(string name){

  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return -1;
  }

  if (!BarcodeList[name].ExtremaFound){
    cout << "Extrema for Barcode "<<name<<" are not constructed!"<<endl;
    return -2;
  }

  if (BarcodeList[name].fType.compare("Reg")!=0){
    cout << "Barcode with name " << name << " is not a regular barcode. Funciton FindBigGaps does not apply." << endl;
    return -3;
  }

  auto NExtrema = BarcodeList[name].fExtremaList.size();
  int NIntervals = 0;
  BarcodeList[name].fBigGapList.clear();
  float tot = 0;

  for (size_t i=0; i<NExtrema-1; i++){
    double interval = BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[i+1])-BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[i]);
    tot += interval;
    interval = 0;
    ++ NIntervals;
  }

  float average = tot/static_cast<float>(NIntervals);
  cout << "average: " << average << endl;

  for(size_t j=0; j<NExtrema-2; j++){
    double check_interval = BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[j+1])-BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[j]);
    double check_interval2 = BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[j+2])-BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[j+1]);

    if(check_interval > 2*average && check_interval2 > 2*average){
      BarcodeList[name].fBigGapList.push_back(BarcodeList[name].fXTime->at(BarcodeList[name].fExtremaList[j]));
    }
  }
  for(size_t k=0; k<BarcodeList[name].fBigGapList.size(); k++){
    cout << "Big Gap Position: " << BarcodeList[name].fBigGapList[k] << endl;
  }

  return 0;
}
/**********************************************************************/
int gm2BarcodeAnalyzer::ConvertToLogic(string name)
{
  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return -1;
  }
  //If extrema are not found, find them first
  if (!BarcodeList[name].ExtremaFound){
    FindExtrema(name);
  }
  //Initialize
  BarcodeList[name].fLogicLevels.clear();
  BarcodeList[name].fLogicLevels.resize(BarcodeList[name].fNExtrema,gm2field::LogicLevel{-1,0,0}); //Initialize to -1 for safety check
  BarcodeList[name].fNHighLevels = 0;
  BarcodeList[name].fNLowLevels = 0;

  //Convert analog to lotic levels
  int Level = 0;
  size_t IndexLow = 0;
  size_t IndexHigh = 0;
  //Treat first incomplete section
  IndexLow = 0;
//  IndexHigh = (BarcodeList[name].fMaxList[0]+BarcodeList[name].fMinList[0])/2;
  if (BarcodeList[name].fExtremaList[0] == BarcodeList[name].fMinList[0]){
    IndexHigh = FindHalfRise(name,BarcodeList[name].fExtremaList[0],BarcodeList[name].fExtremaList[1]);
    Level = 0;
    BarcodeList[name].fNLowLevels++;
  }else if (BarcodeList[name].fExtremaList[0] == BarcodeList[name].fMaxList[0]){
    IndexHigh = FindHalfFall(name,BarcodeList[name].fExtremaList[0],BarcodeList[name].fExtremaList[1]);
    Level = 1;
    BarcodeList[name].fNHighLevels++;
  }else{
    cout << "Encountered mismatching of Max/Min list and Extrema list."<<endl;
    return -1;
  }
  BarcodeList[name].fLogicLevels[0] = gm2field::LogicLevel{Level,IndexLow,IndexHigh};
/*  for (size_t i=IndexLow;i<IndexHigh;i++){
    BarcodeList[name].fLogicLevels[i]=Level;
  }*/
  //Alternating Level
  if (Level==0)Level=1;
  else if (Level==1)Level=0;
  //Treat normal section
  for (size_t i=1;i<BarcodeList[name].fNExtrema-1;i++){
    if (Level==0){
      IndexLow = FindHalfFall(name,BarcodeList[name].fExtremaList[i-1],BarcodeList[name].fExtremaList[i]);
      IndexHigh = FindHalfRise(name,BarcodeList[name].fExtremaList[i],BarcodeList[name].fExtremaList[i+1]);
      BarcodeList[name].fNLowLevels++;
    }else if(Level==1){
      IndexLow = FindHalfRise(name,BarcodeList[name].fExtremaList[i-1],BarcodeList[name].fExtremaList[i]);
      IndexHigh = FindHalfFall(name,BarcodeList[name].fExtremaList[i],BarcodeList[name].fExtremaList[i+1]);
      BarcodeList[name].fNHighLevels++;
    }
    if (IndexHigh>=BarcodeList[name].fNPoints){
      cout << "Error! LogicLevels out of range!"<<endl;
      return -1;
    }
    BarcodeList[name].fLogicLevels[i] = gm2field::LogicLevel{Level,IndexLow,IndexHigh};
    /*
    for (size_t i=IndexLow;i<IndexHigh;i++){
      BarcodeList[name].fLogicLevels[i]=Level;
    }*/
    if (Level==0)Level=1;
    else if (Level==1)Level=0;
  }
  //Treat last step
  IndexHigh = BarcodeList[name].fNPoints-1;
  if (Level==0){
    IndexLow = FindHalfFall(name,BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-2],BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-1]);
    BarcodeList[name].fNLowLevels++;
  }else if(Level==1){
    IndexLow = FindHalfRise(name,BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-2],BarcodeList[name].fExtremaList[BarcodeList[name].fNExtrema-1]);
    BarcodeList[name].fNHighLevels++;
  }
  BarcodeList[name].fLogicLevels[BarcodeList[name].fNExtrema-1] = gm2field::LogicLevel{Level,IndexLow,IndexHigh};
/*  for (size_t i=IndexLow;i<=IndexHigh;i++){
    BarcodeList[name].fLogicLevels[i]=Level;
  }*/

  //Testing and throwing out warnings and errors
  if (BarcodeList[name].fNExtrema!=(BarcodeList[name].fNHighLevels+BarcodeList[name].fNLowLevels)){
    cout <<"Warning! BarcodeList[name].fNExtrema != BarcodeList[name].fNHighLevels+BarcodeList[name].fNLowLevels. BarcodeList[name].fNExtrema="<<BarcodeList[name].fNExtrema<<" BarcodeList[name].fNHighLevels+BarcodeList[name].fNLowLevels="<<BarcodeList[name].fNHighLevels+BarcodeList[name].fNLowLevels<<endl;
  }
  /*
  for (size_t i=0;i<BarcodeList[name].fNPoints;i++){
    if (BarcodeList[name].fLogicLevels[i]==-1){
      cout <<"Error! LogicLevels is not completely filled at "<<i<<endl;
      return -1;
    }
  }*/
  //Try using lambda
  auto count = count_if(begin(BarcodeList[name].fLogicLevels), end(BarcodeList[name].fLogicLevels), [](gm2field::LogicLevel i){ return i.Level==-1; } );
  if (count>0){
    cout <<"Error! LogicLevels is not completely filled"<<endl;
    return -1;
  }
  BarcodeList[name].LogicLevelConverted = true;

  return 0;
}

/**********************************************************************/
int gm2BarcodeAnalyzer::AnalyzeAll()
{
  for (std::map<string,TBarcode>::iterator it=BarcodeList.begin(); it!=BarcodeList.end(); ++it){
    FindExtrema(it->first);
    ConvertToLogic(it->first);
  }
  ChopSegments("Abs1","Pos1");
  Decode("Abs1");
  ChopSegments("Abs2","Pos2");
  Decode("Abs2");

  return 0;
}
/*********************************************************************/
void gm2BarcodeAnalyzer::Smooth(string name, size_t PointsAveraged){
  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return;
  }
  float tot = 0;  //Used to sum y values that will be averaged
  vector<float> temp_vec;
  float average = 0;

  for (size_t k = 0; k<BarcodeList[name].fNPoints; ++k){

    if(k<PointsAveraged/2){ //Handle the first points
      temp_vec.push_back(BarcodeList[name].fY->at(k));
      continue;
    }

    if(k>=BarcodeList[name].fNPoints-(PointsAveraged/2)){ //Handle the last points
      temp_vec.push_back(BarcodeList[name].fY->at(k));
      continue;
    }

    for(size_t i=(k-PointsAveraged/2); i<=(k+PointsAveraged/2); i++){
      tot += BarcodeList[name].fY->at(i);
    }
    average = tot/PointsAveraged;
    temp_vec.push_back(average);

    tot = 0;	
  }	

  for(size_t j=0; j<BarcodeList[name].fNPoints; ++j){
    BarcodeList[name].fY->at(j) = temp_vec[j];
  }
  temp_vec.clear();
}
/**********************************************************************/
//AbsBarcode
/**********************************************************************/
int gm2BarcodeAnalyzer::ChopSegments(string name, string reference)
{
  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return -1;
  }
  if (BarcodeList.find(reference) == BarcodeList.end()){
    cout << "Barcode with name " << reference << " is not yet registered." << endl;
    return -1;
  }
  const TBarcode& RefReg = BarcodeList[reference];
  //Clean up the previous segmentations
  if (BarcodeList[name].fSegmented){
    BarcodeList[name].fSegmentList.clear();
    BarcodeList[name].fAuxList.clear();
    BarcodeList[name].fSegmented=false;
  }
  //Check if the logic levels are converted
  if (!BarcodeList[name].LogicLevelConverted){
    ConvertToLogic(name);
  }
  if (!RefReg.LogicLevelConverted){
    cout <<"Error! Reference RegBarcode is not yet converted to Logic levels."<<endl;
    return -1;
  }
  BarcodeList[name].fAuxList.resize(BarcodeList[name].fNExtrema);

  //Get Logic level list from RefReg
  auto RefLevels = RefReg.fLogicLevels;
  auto NRefLevels = RefLevels.size();
  size_t i{0};
  size_t j{0};
  while (i<BarcodeList[name].fNExtrema && j<NRefLevels){
    auto LeftBound = BarcodeList[name].fLogicLevels[i].LEdge;
    auto RightBound = BarcodeList[name].fLogicLevels[i].REdge;
    while (j<NRefLevels){
      if (RefLevels[j].REdge<LeftBound){
	j++;
	continue;
      }
      else{
	BarcodeList[name].fAuxList[i].push_back(RefLevels[j]);
	if (RefLevels[j].REdge<RightBound){
	  j++;
	  continue;
	}else if (RefLevels[j].REdge==RightBound){
	  if (j+1<NRefLevels)BarcodeList[name].fAuxList[i].push_back(RefLevels[j+1]);
	  break;
	}
	else{
	  break;
	}
      }
    }
    i++;
  }
  
  //Constructs segments
  i=0;
  while (i<BarcodeList[name].fNExtrema){
    size_t NLevelsRef = BarcodeList[name].fAuxList[i].size();
    if (NLevelsRef>=16 && NLevelsRef<=20){
      BarcodeList[name].fSegmentList.push_back(gm2field::AbsBarcodeSegment{0,-1,1,NLevelsRef,vector<size_t>{i},BarcodeList[name].fAuxList[i]});
      i++;
      continue;
    }else{
      vector<size_t> TempAbsList;
      vector<gm2field::LogicLevel> TempRegList;
      while (i<BarcodeList[name].fNExtrema){
	size_t NLevelsRef2 = BarcodeList[name].fAuxList[i].size();
	if (NLevelsRef2>=16 && NLevelsRef2<=20)break;
	TempAbsList.push_back(i);
	for (j=0;j<BarcodeList[name].fAuxList[i].size();j++){
	  if (TempRegList.size()!=0){
	    if ((TempRegList.back().LEdge==BarcodeList[name].fAuxList[i][j].LEdge) && (TempRegList.back().REdge==BarcodeList[name].fAuxList[i][j].REdge)){
	      continue;
	    }
	    //Also need to check second-last, may overlap in those cases where edges are overlaping
	    if ((TempRegList[TempRegList.size()-2].LEdge==BarcodeList[name].fAuxList[i][j].LEdge) && (TempRegList[TempRegList.size()-2].REdge==BarcodeList[name].fAuxList[i][j].REdge)){
	      continue;
	    }
	  }
	  TempRegList.push_back(BarcodeList[name].fAuxList[i][j]);
	}
	i++;
      }
      BarcodeList[name].fSegmentList.push_back(gm2field::AbsBarcodeSegment{1,-1,TempAbsList.size(),TempRegList.size(),TempAbsList,TempRegList}); //For the moment, lable all non-trivial region coded, later the list will be scanned again to rule out abnormals
//      cout << "Temp Reg List Size: " << TempRegList.size() << endl;
    }
  }

  BarcodeList[name].fNSegments = BarcodeList[name].fSegmentList.size();
  //Scan SegmentList, mark all too long or too short regions abnormal
  for (size_t i=0; i<BarcodeList[name].fNSegments; i++){
    if(BarcodeList[name].fSegmentList[i].RegionType==0){
      continue;
    }
    auto RegLevelList = BarcodeList[name].fSegmentList[i].fRegLevelList;
    if (RegLevelList.size()>14 || RegLevelList.size()<13){
      BarcodeList[name].fSegmentList[i].RegionType=-1;
    }
  }
  BarcodeList[name].fSegmented=true;
  return 0;
}

/**********************************************************************/
int gm2BarcodeAnalyzer::Decode(string name)
{
  if (BarcodeList.find(name) == BarcodeList.end()){
    cout << "Barcode with name " << name << " is not yet registered." << endl;
    return -1;
  }
  if (!BarcodeList[name].fSegmented){
    cout << "Abs Barcode "<<name<<" is not segmented! Decode needs segmentation."<<endl;
    return -1;
  }
  for (size_t i=0; i<BarcodeList[name].fNSegments; ++i){
    if(BarcodeList[name].fSegmentList[i].RegionType!=1){
      continue;
    }
    //Determine Direction
    int indexLEdge = BarcodeList[name].fLogicLevels[BarcodeList[name].fSegmentList[i].fLevelIndexList.front()].LEdge;
    int indexREdge = BarcodeList[name].fLogicLevels[BarcodeList[name].fSegmentList[i].fLevelIndexList.back()].REdge;
    int LDir = 0;
    int RDir = 0;
    if (indexLEdge == 0 ){
      LDir = 0;
    }else{
      if (BarcodeList[name].fXPos->at(indexLEdge) - BarcodeList[name].fXPos->at(indexLEdge-1) > 0) LDir = 1;
      if (BarcodeList[name].fXPos->at(indexLEdge) - BarcodeList[name].fXPos->at(indexLEdge-1) < 0) LDir = -1;
      if (BarcodeList[name].fXPos->at(indexLEdge) - BarcodeList[name].fXPos->at(indexLEdge-1) == 0) LDir = 0;
    }
    if (indexREdge == 0 ){
      RDir = 0;
    }else{
      if (BarcodeList[name].fXPos->at(indexREdge) - BarcodeList[name].fXPos->at(indexREdge-1) > 0) RDir = 1;
      if (BarcodeList[name].fXPos->at(indexREdge) - BarcodeList[name].fXPos->at(indexREdge-1) < 0) RDir = -1;
      if (BarcodeList[name].fXPos->at(indexREdge) - BarcodeList[name].fXPos->at(indexREdge-1) == 0) RDir = 0;
    }
    double TimeStamp = BarcodeList[name].fXTime->at(BarcodeList[name].fLogicLevels[BarcodeList[name].fSegmentList[i].fLevelIndexList.front()].LEdge);
    if (LDir!=RDir){
      continue;
    }else if(LDir==0 || RDir==0){
      continue;
    }
    auto AbsIndexList = BarcodeList[name].fSegmentList[i].fLevelIndexList;
    auto RegLevelList = BarcodeList[name].fSegmentList[i].fRegLevelList;
    if (RegLevelList.size()==14){
      RegLevelList.erase(RegLevelList.begin()); //Remove first element
      RegLevelList.pop_back();                  //Remove last element
    }else if (RegLevelList.size()==13){
      if(LDir==1 && RDir==1){
	if (RegLevelList.front().Level==0)RegLevelList.erase(RegLevelList.begin());
	if (RegLevelList.back().Level==1)RegLevelList.pop_back();
      }else if(LDir==-1 && RDir==-1){
	if (RegLevelList.front().Level==1)RegLevelList.erase(RegLevelList.begin());
	if (RegLevelList.back().Level==0)RegLevelList.pop_back();
      }
    }else{
      cout <<"Error. Code region length is not normal. Length = "<<RegLevelList.size()<<endl;
      continue;//In bad region, end region or something else
//      return -1;
    }
    vector<int> DigitMap(RegLevelList.size(),-1); //Temporarily save the digits from the absolute barcode

    for (size_t j=0; j<RegLevelList.size(); ++j){
      auto RegLevelUnit = RegLevelList[j];
      for (size_t k=0; k<AbsIndexList.size(); ++k){
	auto AbsLevelIndex = AbsIndexList[k];
	auto AbsLevelUnit = BarcodeList[name].fLogicLevels[AbsLevelIndex];

	if (RegLevelUnit.REdge <= AbsLevelUnit.LEdge){
	  continue;
	}else if(AbsLevelUnit.REdge <= RegLevelUnit.LEdge){
	  continue;
	}else if(RegLevelUnit.LEdge <= AbsLevelUnit.LEdge && AbsLevelUnit.REdge <= RegLevelUnit.REdge){
	  DigitMap[j] = AbsLevelUnit.Level;
	  break;
	}else if(RegLevelUnit.LEdge >= AbsLevelUnit.LEdge && AbsLevelUnit.REdge >= RegLevelUnit.REdge){
	  DigitMap[j] = AbsLevelUnit.Level;
	  break;
	}else if(RegLevelUnit.LEdge < AbsLevelUnit.LEdge && RegLevelUnit.REdge > AbsLevelUnit.LEdge && RegLevelUnit.REdge < AbsLevelUnit.REdge){
	  double OverlapRatio = (BarcodeList[name].fXTime->at(RegLevelUnit.REdge)-BarcodeList[name].fXTime->at(AbsLevelUnit.LEdge))/(BarcodeList[name].fXTime->at(RegLevelUnit.REdge)-BarcodeList[name].fXTime->at(RegLevelUnit.LEdge));
	  if (OverlapRatio < 0.5){
	    continue;
	  }
	  if (OverlapRatio >= 0.5){
	    DigitMap[j] = AbsLevelUnit.Level;
	    break;
	  }
	}else if(RegLevelUnit.LEdge > AbsLevelUnit.LEdge && RegLevelUnit.LEdge < AbsLevelUnit.REdge && RegLevelUnit.REdge > AbsLevelUnit.REdge){
	  double OverlapRatio = (BarcodeList[name].fXTime->at(AbsLevelUnit.REdge)-BarcodeList[name].fXTime->at(RegLevelUnit.LEdge))/(BarcodeList[name].fXTime->at(RegLevelUnit.REdge)-BarcodeList[name].fXTime->at(RegLevelUnit.LEdge));
	  if (OverlapRatio < 0.5){
	    continue;
	  }
	  if (OverlapRatio >= 0.5){
	    DigitMap[j] = AbsLevelUnit.Level;
	    break;
	  }
	}else{
	  cout << "Error! in Decode Function an exception case occured." << endl;
	  cout << "Regular Left: " << BarcodeList[name].fXTime->at(RegLevelUnit.LEdge) << endl;
	  cout << "Regular Right: " << BarcodeList[name].fXTime->at(RegLevelUnit.REdge) << endl;
	  cout << "Absolute Left: " << BarcodeList[name].fXTime->at(AbsLevelUnit.LEdge) << endl;
	  cout << "Absolute Right: " << BarcodeList[name].fXTime->at(AbsLevelUnit.REdge) << endl;
	  return -1;
	}

      }
    }
    DigitMap.erase(DigitMap.begin()); //Delete first element
    DigitMap.pop_back();              //Delete last element
    size_t DigitMapSize = DigitMap.size();
//    cout << "Digit Map Size: " << DigitMapSize << " Reg Level List Size: " << RegLevelList.size() << endl;
    /*  	vector<int> pos_vec;
		for(unsigned int m=0; m<DigitMap.size(); m++){
		cout << DigitMap[m] << " ";
		if(DigitMap[m] == 0){
		continue;
		}else if (DigitMap[m] == 1){
		pos_vec.push_back(m);
		}
		}*/

    //Make complement, change direction if moving CW
    if(LDir==-1 && RDir==-1){
      for(size_t n=0; n<DigitMapSize/2; n++){
	int temp = DigitMap[n];
	DigitMap[n] = DigitMap[DigitMapSize-1-n];
	DigitMap[DigitMapSize-1-n] = temp;
      }
    }
    for(size_t n=0; n<DigitMapSize; n++){
      if(DigitMap[n] == 0){
	DigitMap[n] = 1;
      }else if(DigitMap[n] == 1){
	DigitMap[n] = 0;
      }else{
	cout <<"Error! Missing one or more interpretations of digits in Abs Barcode."<<endl;
	return -1;
      }
    }
    //Convert binary to digital
    if (DigitMapSize!=10){//In bad region, end region or something else
      continue;
    }
    int DecimalNumber = 0;
    cout << "TimeStamp "<< TimeStamp<< " ";
    cout <<"Binary code: ";
    for(size_t j=0; j<DigitMapSize; j++){
      cout <<DigitMap[j];
      DecimalNumber += DigitMap[j] << j;
    } 
    cout <<" ; ";
    cout << "Decimal Number: " << DecimalNumber<< endl;;
  }

  return 0;
}

