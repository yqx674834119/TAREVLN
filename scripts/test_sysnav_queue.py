"""ASan regression of the actual native selection method with small shrinking queues."""
import argparse
import hashlib
from pathlib import Path
import subprocess
ap=argparse.ArgumentParser()
ap.add_argument('source',type=Path)
ap.add_argument('output',type=Path)
args=ap.parse_args()
source=(args.source/'src/exploration_planner/tare_planner/src/local_coverage_planner/local_coverage_planner.cpp').read_text()
a=source.index('  void LocalCoveragePlanner::SelectViewPoint(')
b=source.index('  void LocalCoveragePlanner::SelectViewPointFromFrontierQueue(',a)
method=source[a:b]
fixture=r'''
#include <algorithm>
#include <cassert>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include <iostream>
#define MY_ASSERT(x) assert(x)
namespace misc_utils_ns {template<class T> bool InRange(const std::vector<T>& v,int i){return i>=0&&i<int(v.size());}}
bool SortPairInRev(const std::pair<int,int>& a,const std::pair<int,int>& b){return a.first>b.first;}
struct Viewpoints {
 std::vector<std::vector<int>> lists;
 int GetViewPointArrayInd(int i){return i;}
 const std::vector<int>& GetViewPointCoveredPointList(int i,bool){return lists.at(i);}
 const std::vector<int>& GetViewPointCoveredFrontierPointList(int i,bool){return lists.at(i);}
};
struct LocalCoveragePlanner {
 struct {int kMinAddFrontierPointNum=1,kMinAddPointNum=1,kGreedyViewPointSampleRange=3;} parameters_;
 std::shared_ptr<Viewpoints> viewpoint_manager_=std::make_shared<Viewpoints>();
 void SelectViewPoint(const std::vector<std::pair<int,int>>&,const std::vector<bool>&,std::vector<int>&,bool);
};
'''
main=r'''
int main(){
 for(int trial=0;trial<10000;++trial){
  LocalCoveragePlanner planner;
  const int n=1+trial%5;
  std::vector<std::pair<int,int>> queue;
  for(int i=0;i<n;++i){planner.viewpoint_manager_->lists.push_back({3*i,3*i+1,3*i+2});queue.emplace_back(3,i);}
  std::vector<int> selected;
  planner.SelectViewPoint(queue,std::vector<bool>(3*n,false),selected,trial%2);
  assert(selected.size()==size_t(n));
  assert(std::set<int>(selected.begin(),selected.end()).size()==size_t(n));
 }
 std::cout<<"10000 shrinking-queue trials passed (coverage and frontier modes)\n";
}
'''
args.output.mkdir(parents=True,exist_ok=True)
file=args.output/'queue-regression.cpp';file.write_text(fixture+method+main)
binary=args.output/'queue-regression'
subprocess.run(['g++','-std=c++17','-O1','-g','-fsanitize=address','-fno-omit-frame-pointer',str(file),'-o',str(binary)],check=True)
subprocess.run([str(binary)],check=True)
print('native_method_sha256',hashlib.sha256(method.encode()).hexdigest())
