#include <bits/stdc++.h>
using namespace std;
#define rep(i, a) for (int i = 0; i < (int)(a); i++)
#define sz(x) (int)(x).size()
#define pcnt __builtin_popcountll
typedef long long ll;
template<typename T>istream& operator>>(istream&i,vector<T>&v){rep(j,sz(v))i>>v[j];return i;}
template<typename T>string join(const vector<T>&v){stringstream s;rep(i,sz(v))s<<' '<<v[i];return s.str().substr(1);}
template<typename T>ostream& operator<<(ostream&o,const vector<T>&v){if(sz(v))o<<join(v);return o;}
template<typename T1,typename T2>istream& operator>>(istream&i,pair<T1,T2>&v){return i>>v.first>>v.second;}
template<typename T1,typename T2>ostream& operator<<(ostream&o,const pair<T1,T2>&v){return o<<v.first<<","<<v.second;}
template<typename T>bool mins(T& x,const T&y){if(x>y){x=y;return true;}else return false;}
template<typename T>bool maxs(T& x,const T&y){if(x<y){x=y;return true;}else return false;}
template<typename T>T dup(T x, T y){return (x+y-1)/y;}
template<typename T>ll suma(const vector<T>&a){ll res(0);for(auto&&x:a)res+=x;return res;}
int keta(ll n) { int ret = 0; while (n>0) { n/=10; ret++; } return ret; }

#ifdef _DEBUG
inline void dump() { cerr << endl; }
template <typename Head> void dump(Head &&head) { cerr << head; dump(); }
template <typename Head, typename... Tail> void dump(Head &&head, Tail &&... tail) { cerr << head << ", "; dump(forward<Tail>(tail)...); }
#define debug(...) do { cerr << __LINE__ << ":\t" << #__VA_ARGS__ << " = "; dump(__VA_ARGS__); } while (false)
#else
#define dump(...)
#define debug(...)
#endif

template <typename T> struct edge {
  int src, to;
  T cost;
  edge(int to, T cost) : src(-1), to(to), cost(cost) {}
  edge(int src, int to, T cost) : src(src), to(to), cost(cost) {}
  edge &operator=(const int &x) {
    to = x;
    return *this;
  }
  operator int() const { return to; }
};
template <typename T> using Edges = vector<edge<T>>;
template <typename T> using WeightedGraph = vector<Edges<T>>;
using UnWeightedGraph = vector<vector<int>>;
template <typename T> using Matrix = vector<vector<T>>;

const ll LINF = 1LL << 60;
const int INF = 1001001001;

/////////////////////////////////////////////////////////////////////

using P = pair<int, int>;


/////////////////////////////////////////////////////////////////////
//
//  方針
//  1. 犬を一網打尽にする
//    犬は人にまとわり付きがちなので、他の動物を捕獲する際にも邪魔になる
//    1.1. 真ん中に仕掛けを作る
//      1.1.1. 最小費用流を用いて4箇所の地形作成開始ポイントにそれぞれ1人ずつ割り当てる
//            余った人は再び最小費用流で4箇所のいずれかの
//            待機ポイントに割り当てる(全員割当てが決まるまで繰り返し)
//      1.1.2. 1.1.1.で選ばれた人が地形を作成していき、待機ポイントに移動
//    1.2 待機ポイントで全ての人を待たせて仕掛けの中に全ての犬が入り込むのを待つ
//    1.3 良きタイミングで全員一歩仕掛けから離れて(離れる必要がない場合はそのまま？)封鎖する
//  2. 他の動物を捕獲する仕掛けを作成する
//    1.で分けた4グループでそれぞれ行動させるイメージ
//  3. 残りの動物をできるだけ捕獲する
//
/////////////////////////////////////////////////////////////////////

const int H = 30;
const int W = 30;

int turn = 300;

const int PET = 100;
const int HUM = 200;

int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

// 全体のフェーズ管理
int phase = 0;
const int PHASE_0 = 0;      // 初期状態
const int PHASE_1 = 1;      //
const int PHASE_2 = 2;      //
const int PHASE_3 = 3;      //
const int PHASE_4 = 4;      //
const int PHASE_5 = 5;      //
const int PHASE_8 = 8;      //
const int PHASE_9 = 9;      //
const int PHASE_10 = 10;    //
const int FINAL_OPERATION_TURN = 30;

// 回答用文字列
string act = "";
const char ACT_NONE       = '.';  // 何もしない
const char ACT_STOP_UP    = 'u';  // 上のマスを通行不能にする
const char ACT_STOP_DOWN  = 'd';  // 下のマスを通行不能にする
const char ACT_STOP_LEFT  = 'l';  // 左のマスを通行不能にする
const char ACT_STOP_RIGHT = 'r';  // 右のマスを通行不能にする
const char ACT_MOVE_UP    = 'U';  // 上のマスに移動する
const char ACT_MOVE_DOWN  = 'D';  // 下のマスに移動する
const char ACT_MOVE_LEFT  = 'L';  // 左のマスに移動する
const char ACT_MOVE_RIGHT = 'R';  // 右のマスに移動する
const vector<char> ACT_STOP = {
  ACT_STOP_UP, ACT_STOP_DOWN, ACT_STOP_LEFT, ACT_STOP_RIGHT
};
const vector<char> ACT_MOVE = {
  ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT
};

// 各マスの情報
//   0 : 何もない、通行可能
// < 0 : 通行止め
// > 0 : 隣に動物が存在する
vector<vector<bool>> blocked(H, vector<bool>(W, false));

// 設計図
vector<vector<int>> base = {
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 8, 9, 6, 6, 6, 6},
  {9, 9, 6, 9, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 5, 6, 9, 9, 9, 9},
  {9, 9, 9, 6, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 8, 5, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 0, 6, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 6, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 5, 10, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 5, 7, 7, 7, 7, 7, 7, 10, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 10, 7, 7, 7, 7, 7, 7, 10, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 10, 7, 7, 7, 7, 7, 7, 10, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 10, 7, 7, 7, 7, 7, 7, 10, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 10, 7, 7, 7, 7, 7, 7, 10, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 10, 7, 7, 7, 7, 7, 7, 5, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 10, 5, 10, 10, 10, 10, 10, 10, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 6, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 6, 0, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 5, 8, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 6, 9, 9, 9},
  {9, 9, 9, 9, 6, 5, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 9, 6, 9, 9},
  {6, 6, 6, 6, 9, 8, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9}
};

vector<vector<int>> base_few_dogs = {
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 8, 9, 6, 6, 6, 6},
  {9, 9, 6, 9, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 5, 6, 9, 9, 9, 9},
  {9, 9, 9, 6, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 8, 5, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 0, 6, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 6, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 6, 9, 6, 9, 9, 6, 9, 6, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 9, 10, 10, 10, 10, 5, 10, 9, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 6, 5, 7, 7, 7, 7, 10, 6, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 9, 10, 7, 7, 7, 7, 10, 9, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 9, 10, 7, 7, 7, 7, 10, 9, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 6, 10, 7, 7, 7, 7, 5, 6, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 9, 10, 5, 10, 10, 10, 10, 9, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 6, 9, 6, 9, 9, 6, 9, 6, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 6, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 6, 0, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 5, 8, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 6, 9, 9, 9},
  {9, 9, 9, 9, 6, 5, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 9, 6, 9, 9},
  {6, 6, 6, 6, 9, 8, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9}
};

vector<vector<int>> base_nodog = {
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 9, 9, 9, 9},
  {9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 8, 9, 6, 6, 6, 6},
  {9, 9, 6, 9, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 5, 6, 9, 9, 9, 9},
  {9, 9, 9, 6, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 8, 5, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 0, 6, 5, 5, 0, 6, 5, 5, 0, 6, 5, 5, 0, 0, 0, 0, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 8, 8, 6, 9, 8, 8, 6, 9, 8, 8, 6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 6, 9, 9, 9, 9, 6, 6, 6, 9, 6, 6, 6, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 6, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 6, 9, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 9, 6, 6, 6, 9, 9, 9, 9, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 2, 5, 8, 9, 9, 9, 9, 9, 9, 9, 6, 6, 6, 9, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 9, 6, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 6, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 0, 6, 6, 6, 9, 6, 6, 6, 9, 9, 9, 9, 6, 9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 9, 6, 6, 6, 6, 0, 2, 5, 9, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 5, 8, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 9, 9, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 6, 6, 6, 6, 8, 8, 9, 6, 8, 8, 9, 6, 8, 8, 9, 6, 0, 2, 5, 8, 9, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 0, 0, 0, 0, 5, 5, 6, 0, 5, 5, 6, 0, 5, 5, 6, 0, 0, 1, 0, 6, 6, 6, 6},
  {6, 6, 6, 6, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 5, 8, 9, 9},
  {9, 9, 9, 8, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 6, 9, 9, 9},
  {9, 9, 9, 9, 6, 5, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 9, 6, 9, 9},
  {6, 6, 6, 6, 9, 8, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9},
  {9, 9, 9, 9, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9, 6, 9, 9}
};

vector<pair<P, char>> build_sequence_periphery = {
  {{ 3,  3}, ACT_STOP_DOWN},
  {{ 3,  3}, ACT_STOP_LEFT},
  {{ 3,  3}, ACT_MOVE_UP},
  {{ 2,  3}, ACT_STOP_LEFT},
  {{ 2,  3}, ACT_MOVE_UP},
  {{ 1,  3}, ACT_STOP_LEFT},
  {{ 1,  3}, ACT_MOVE_UP},
  {{ 0,  3}, ACT_STOP_LEFT},
  {{ 0,  3}, ACT_MOVE_RIGHT},
  {{ 0,  4}, ACT_MOVE_RIGHT},
  {{ 0,  5}, ACT_MOVE_RIGHT},
  {{ 0,  6}, ACT_STOP_LEFT},
  {{ 0,  6}, ACT_MOVE_DOWN},
  {{ 1,  6}, ACT_STOP_LEFT},
  {{ 1,  6}, ACT_MOVE_DOWN},
  {{ 2,  6}, ACT_STOP_LEFT},
  {{ 2,  6}, ACT_MOVE_DOWN},
  {{ 3,  6}, ACT_STOP_LEFT},
  {{ 3,  6}, ACT_MOVE_RIGHT},
  {{ 3,  7}, ACT_MOVE_RIGHT},
  {{ 3,  8}, ACT_MOVE_RIGHT},
  {{ 3,  9}, ACT_STOP_LEFT},
  {{ 3,  9}, ACT_MOVE_UP},
  {{ 2,  9}, ACT_STOP_LEFT},
  {{ 2,  9}, ACT_MOVE_UP},
  {{ 1,  9}, ACT_STOP_LEFT},
  {{ 1,  9}, ACT_MOVE_UP},
  {{ 0,  9}, ACT_STOP_LEFT},
  {{ 0,  9}, ACT_MOVE_RIGHT},
  {{ 0, 10}, ACT_MOVE_RIGHT},
  {{ 0, 11}, ACT_MOVE_RIGHT},
  {{ 0, 12}, ACT_STOP_LEFT},
  {{ 0, 12}, ACT_MOVE_DOWN},
  {{ 1, 12}, ACT_STOP_LEFT},
  {{ 1, 12}, ACT_MOVE_DOWN},
  {{ 2, 12}, ACT_STOP_LEFT},
  {{ 2, 12}, ACT_MOVE_DOWN},
  {{ 3, 12}, ACT_STOP_LEFT},
  {{ 3, 12}, ACT_MOVE_RIGHT},
  {{ 3, 13}, ACT_MOVE_RIGHT},
  {{ 3, 14}, ACT_MOVE_RIGHT},
  {{ 3, 15}, ACT_STOP_LEFT},
  {{ 3, 15}, ACT_MOVE_UP},
  {{ 2, 15}, ACT_STOP_LEFT},
  {{ 2, 15}, ACT_MOVE_UP},
  {{ 1, 15}, ACT_STOP_LEFT},
  {{ 1, 15}, ACT_MOVE_UP},
  {{ 0, 15}, ACT_STOP_LEFT},
  {{ 0, 15}, ACT_MOVE_RIGHT},
  {{ 0, 16}, ACT_MOVE_RIGHT},
  {{ 0, 17}, ACT_MOVE_RIGHT},
  {{ 0, 18}, ACT_STOP_LEFT},
  {{ 0, 18}, ACT_MOVE_DOWN},
  {{ 1, 18}, ACT_STOP_LEFT},
  {{ 1, 18}, ACT_MOVE_DOWN},
  {{ 2, 18}, ACT_STOP_LEFT},
  {{ 2, 18}, ACT_MOVE_DOWN},
  {{ 3, 18}, ACT_STOP_LEFT},
  {{ 3, 18}, ACT_MOVE_RIGHT},
  {{ 3, 19}, ACT_MOVE_RIGHT},
  {{ 3, 20}, ACT_MOVE_RIGHT},
  {{ 3, 21}, ACT_STOP_LEFT},
  {{ 3, 21}, ACT_MOVE_UP},
  {{ 2, 21}, ACT_STOP_LEFT},
  {{ 2, 21}, ACT_MOVE_UP},
  {{ 1, 21}, ACT_STOP_LEFT},
  {{ 1, 21}, ACT_MOVE_UP},
  {{ 0, 21}, ACT_STOP_LEFT},
  {{ 0, 21}, ACT_MOVE_RIGHT},
  {{ 0, 22}, ACT_MOVE_RIGHT},
  {{ 0, 23}, ACT_MOVE_RIGHT},
  {{ 0, 24}, ACT_STOP_LEFT},
  {{ 0, 24}, ACT_MOVE_DOWN},
  {{ 1, 24}, ACT_STOP_LEFT},
  {{ 1, 24}, ACT_MOVE_DOWN},
  {{ 2, 24}, ACT_STOP_LEFT},
  {{ 2, 24}, ACT_MOVE_DOWN},
  {{ 3, 24}, ACT_STOP_LEFT},
  {{ 3, 24}, ACT_MOVE_RIGHT},
  {{ 3, 25}, ACT_MOVE_RIGHT},
  {{ 3, 26}, ACT_STOP_LEFT},
  {{ 3, 26}, ACT_STOP_UP},
  {{ 3, 26}, ACT_MOVE_RIGHT},
  {{ 3, 27}, ACT_STOP_UP},
  {{ 3, 27}, ACT_MOVE_RIGHT},
  {{ 3, 28}, ACT_STOP_UP},
  {{ 3, 28}, ACT_MOVE_RIGHT},
  {{ 3, 29}, ACT_STOP_UP},
  {{ 3, 29}, ACT_MOVE_DOWN},
  {{ 4, 29}, ACT_MOVE_DOWN},
  {{ 5, 29}, ACT_MOVE_DOWN},
  {{ 6, 29}, ACT_STOP_UP},
  {{ 6, 29}, ACT_MOVE_LEFT},
  {{ 6, 28}, ACT_STOP_UP},
  {{ 6, 28}, ACT_MOVE_LEFT},
  {{ 6, 27}, ACT_STOP_UP},
  {{ 6, 27}, ACT_MOVE_LEFT},
  {{ 6, 26}, ACT_STOP_UP},
  {{ 6, 26}, ACT_MOVE_DOWN},
  {{ 7, 26}, ACT_MOVE_DOWN},
  {{ 8, 26}, ACT_MOVE_DOWN},
  {{ 9, 26}, ACT_STOP_UP},
  {{ 9, 26}, ACT_MOVE_RIGHT},
  {{ 9, 27}, ACT_STOP_UP},
  {{ 9, 27}, ACT_MOVE_RIGHT},
  {{ 9, 28}, ACT_STOP_UP},
  {{ 9, 28}, ACT_MOVE_RIGHT},
  {{ 9, 29}, ACT_STOP_UP},
  {{ 9, 29}, ACT_MOVE_DOWN},
  {{10, 29}, ACT_MOVE_DOWN},
  {{11, 29}, ACT_MOVE_DOWN},
  {{12, 29}, ACT_STOP_UP},
  {{12, 29}, ACT_MOVE_LEFT},
  {{12, 28}, ACT_STOP_UP},
  {{12, 28}, ACT_MOVE_LEFT},
  {{12, 27}, ACT_STOP_UP},
  {{12, 27}, ACT_MOVE_LEFT},
  {{12, 26}, ACT_STOP_UP},
  {{12, 26}, ACT_MOVE_DOWN},
  {{13, 26}, ACT_MOVE_DOWN},
  {{14, 26}, ACT_MOVE_DOWN},
  {{15, 26}, ACT_STOP_UP},
  {{15, 26}, ACT_MOVE_RIGHT},
  {{15, 27}, ACT_STOP_UP},
  {{15, 27}, ACT_MOVE_RIGHT},
  {{15, 28}, ACT_STOP_UP},
  {{15, 28}, ACT_MOVE_RIGHT},
  {{15, 29}, ACT_STOP_UP},
  {{15, 29}, ACT_MOVE_DOWN},
  {{16, 29}, ACT_MOVE_DOWN},
  {{17, 29}, ACT_MOVE_DOWN},
  {{18, 29}, ACT_STOP_UP},
  {{18, 29}, ACT_MOVE_LEFT},
  {{18, 28}, ACT_STOP_UP},
  {{18, 28}, ACT_MOVE_LEFT},
  {{18, 27}, ACT_STOP_UP},
  {{18, 27}, ACT_MOVE_LEFT},
  {{18, 26}, ACT_STOP_UP},
  {{18, 26}, ACT_MOVE_DOWN},
  {{19, 26}, ACT_MOVE_DOWN},
  {{20, 26}, ACT_MOVE_DOWN},
  {{21, 26}, ACT_STOP_UP},
  {{21, 26}, ACT_MOVE_RIGHT},
  {{21, 27}, ACT_STOP_UP},
  {{21, 27}, ACT_MOVE_RIGHT},
  {{21, 28}, ACT_STOP_UP},
  {{21, 28}, ACT_MOVE_RIGHT},
  {{21, 29}, ACT_STOP_UP},
  {{21, 29}, ACT_MOVE_DOWN},
  {{22, 29}, ACT_MOVE_DOWN},
  {{23, 29}, ACT_MOVE_DOWN},
  {{24, 29}, ACT_STOP_UP},
  {{24, 29}, ACT_MOVE_LEFT},
  {{24, 28}, ACT_STOP_UP},
  {{24, 28}, ACT_MOVE_LEFT},
  {{24, 27}, ACT_STOP_UP},
  {{24, 27}, ACT_MOVE_LEFT},
  {{24, 26}, ACT_STOP_UP},
  {{24, 26}, ACT_MOVE_DOWN},
  {{25, 26}, ACT_MOVE_DOWN},
  {{26, 26}, ACT_STOP_UP},
  {{26, 26}, ACT_STOP_RIGHT},
  {{26, 26}, ACT_MOVE_DOWN},
  {{27, 26}, ACT_STOP_RIGHT},
  {{27, 26}, ACT_MOVE_DOWN},
  {{28, 26}, ACT_STOP_RIGHT},
  {{28, 26}, ACT_MOVE_DOWN},
  {{29, 26}, ACT_STOP_RIGHT},
  {{29, 26}, ACT_MOVE_LEFT},
  {{29, 25}, ACT_MOVE_LEFT},
  {{29, 24}, ACT_MOVE_LEFT},
  {{29, 23}, ACT_STOP_RIGHT},
  {{29, 23}, ACT_MOVE_UP},
  {{28, 23}, ACT_STOP_RIGHT},
  {{28, 23}, ACT_MOVE_UP},
  {{27, 23}, ACT_STOP_RIGHT},
  {{27, 23}, ACT_MOVE_UP},
  {{26, 23}, ACT_STOP_RIGHT},
  {{26, 23}, ACT_MOVE_LEFT},
  {{26, 22}, ACT_MOVE_LEFT},
  {{26, 21}, ACT_MOVE_LEFT},
  {{26, 20}, ACT_STOP_RIGHT},
  {{26, 20}, ACT_MOVE_DOWN},
  {{27, 20}, ACT_STOP_RIGHT},
  {{27, 20}, ACT_MOVE_DOWN},
  {{28, 20}, ACT_STOP_RIGHT},
  {{28, 20}, ACT_MOVE_DOWN},
  {{29, 20}, ACT_STOP_RIGHT},
  {{29, 20}, ACT_MOVE_LEFT},
  {{29, 19}, ACT_MOVE_LEFT},
  {{29, 18}, ACT_MOVE_LEFT},
  {{29, 17}, ACT_STOP_RIGHT},
  {{29, 17}, ACT_MOVE_UP},
  {{28, 17}, ACT_STOP_RIGHT},
  {{28, 17}, ACT_MOVE_UP},
  {{27, 17}, ACT_STOP_RIGHT},
  {{27, 17}, ACT_MOVE_UP},
  {{26, 17}, ACT_STOP_RIGHT},
  {{26, 17}, ACT_MOVE_LEFT},
  {{26, 16}, ACT_MOVE_LEFT},
  {{26, 15}, ACT_MOVE_LEFT},
  {{26, 14}, ACT_STOP_RIGHT},
  {{26, 14}, ACT_MOVE_DOWN},
  {{27, 14}, ACT_STOP_RIGHT},
  {{27, 14}, ACT_MOVE_DOWN},
  {{28, 14}, ACT_STOP_RIGHT},
  {{28, 14}, ACT_MOVE_DOWN},
  {{29, 14}, ACT_STOP_RIGHT},
  {{29, 14}, ACT_MOVE_LEFT},
  {{29, 13}, ACT_MOVE_LEFT},
  {{29, 12}, ACT_MOVE_LEFT},
  {{29, 11}, ACT_STOP_RIGHT},
  {{29, 11}, ACT_MOVE_UP},
  {{28, 11}, ACT_STOP_RIGHT},
  {{28, 11}, ACT_MOVE_UP},
  {{27, 11}, ACT_STOP_RIGHT},
  {{27, 11}, ACT_MOVE_UP},
  {{26, 11}, ACT_STOP_RIGHT},
  {{26, 11}, ACT_MOVE_LEFT},
  {{26, 10}, ACT_MOVE_LEFT},
  {{26,  9}, ACT_MOVE_LEFT},
  {{26,  8}, ACT_STOP_RIGHT},
  {{26,  8}, ACT_MOVE_DOWN},
  {{27,  8}, ACT_STOP_RIGHT},
  {{27,  8}, ACT_MOVE_DOWN},
  {{28,  8}, ACT_STOP_RIGHT},
  {{28,  8}, ACT_MOVE_DOWN},
  {{29,  8}, ACT_STOP_RIGHT},
  {{29,  8}, ACT_MOVE_LEFT},
  {{29,  7}, ACT_MOVE_LEFT},
  {{29,  6}, ACT_MOVE_LEFT},
  {{29,  5}, ACT_STOP_RIGHT},
  {{29,  5}, ACT_MOVE_UP},
  {{28,  5}, ACT_STOP_RIGHT},
  {{28,  5}, ACT_MOVE_UP},
  {{27,  5}, ACT_STOP_RIGHT},
  {{27,  5}, ACT_MOVE_UP},
  {{26,  5}, ACT_STOP_RIGHT},
  {{26,  5}, ACT_MOVE_LEFT},
  {{26,  4}, ACT_MOVE_LEFT},
  {{26,  3}, ACT_STOP_RIGHT},
  {{26,  3}, ACT_STOP_DOWN},
  {{26,  3}, ACT_MOVE_LEFT},
  {{26,  2}, ACT_STOP_DOWN},
  {{26,  2}, ACT_MOVE_LEFT},
  {{26,  1}, ACT_STOP_DOWN},
  {{26,  1}, ACT_MOVE_LEFT},
  {{26,  0}, ACT_STOP_DOWN},
  {{26,  0}, ACT_MOVE_UP},
  {{25,  0}, ACT_MOVE_UP},
  {{24,  0}, ACT_MOVE_UP},
  {{23,  0}, ACT_STOP_DOWN},
  {{23,  0}, ACT_MOVE_RIGHT},
  {{23,  1}, ACT_STOP_DOWN},
  {{23,  1}, ACT_MOVE_RIGHT},
  {{23,  2}, ACT_STOP_DOWN},
  {{23,  2}, ACT_MOVE_RIGHT},
  {{23,  3}, ACT_STOP_DOWN},
  {{23,  3}, ACT_MOVE_UP},
  {{22,  3}, ACT_MOVE_UP},
  {{21,  3}, ACT_MOVE_UP},
  {{20,  3}, ACT_STOP_DOWN},
  {{20,  3}, ACT_MOVE_LEFT},
  {{20,  2}, ACT_STOP_DOWN},
  {{20,  2}, ACT_MOVE_LEFT},
  {{20,  1}, ACT_STOP_DOWN},
  {{20,  1}, ACT_MOVE_LEFT},
  {{20,  0}, ACT_STOP_DOWN},
  {{20,  0}, ACT_MOVE_UP},
  {{19,  0}, ACT_MOVE_UP},
  {{18,  0}, ACT_MOVE_UP},
  {{17,  0}, ACT_STOP_DOWN},
  {{17,  0}, ACT_MOVE_RIGHT},
  {{17,  1}, ACT_STOP_DOWN},
  {{17,  1}, ACT_MOVE_RIGHT},
  {{17,  2}, ACT_STOP_DOWN},
  {{17,  2}, ACT_MOVE_RIGHT},
  {{17,  3}, ACT_STOP_DOWN},
  {{17,  3}, ACT_MOVE_UP},
  {{16,  3}, ACT_MOVE_UP},
  {{15,  3}, ACT_MOVE_UP},
  {{14,  3}, ACT_STOP_DOWN},
  {{14,  3}, ACT_MOVE_LEFT},
  {{14,  2}, ACT_STOP_DOWN},
  {{14,  2}, ACT_MOVE_LEFT},
  {{14,  1}, ACT_STOP_DOWN},
  {{14,  1}, ACT_MOVE_LEFT},
  {{14,  0}, ACT_STOP_DOWN},
  {{14,  0}, ACT_MOVE_UP},
  {{13,  0}, ACT_MOVE_UP},
  {{12,  0}, ACT_MOVE_UP},
  {{11,  0}, ACT_STOP_DOWN},
  {{11,  0}, ACT_MOVE_RIGHT},
  {{11,  1}, ACT_STOP_DOWN},
  {{11,  1}, ACT_MOVE_RIGHT},
  {{11,  2}, ACT_STOP_DOWN},
  {{11,  2}, ACT_MOVE_RIGHT},
  {{11,  3}, ACT_STOP_DOWN},
  {{11,  3}, ACT_MOVE_UP},
  {{10,  3}, ACT_MOVE_UP},
  {{ 9,  3}, ACT_MOVE_UP},
  {{ 8,  3}, ACT_STOP_DOWN},
  {{ 8,  3}, ACT_MOVE_LEFT},
  {{ 8,  2}, ACT_STOP_DOWN},
  {{ 8,  2}, ACT_MOVE_LEFT},
  {{ 8,  1}, ACT_STOP_DOWN},
  {{ 8,  1}, ACT_MOVE_LEFT},
  {{ 8,  0}, ACT_STOP_DOWN},
  {{ 8,  0}, ACT_MOVE_UP},
  {{ 7,  0}, ACT_MOVE_UP},
  {{ 6,  0}, ACT_MOVE_UP},
  {{ 5,  0}, ACT_STOP_DOWN},
  {{ 5,  0}, ACT_MOVE_RIGHT},
  {{ 5,  1}, ACT_STOP_DOWN},
  {{ 5,  1}, ACT_MOVE_RIGHT},
  {{ 5,  2}, ACT_STOP_DOWN},
  {{ 5,  2}, ACT_MOVE_RIGHT},
  {{ 5,  3}, ACT_STOP_DOWN},
  {{ 5,  3}, ACT_MOVE_UP},
  {{ 4,  3}, ACT_MOVE_UP}
};

vector<pair<P, char>> build_sequence_center = {
  {{11, 10}, ACT_STOP_UP},
  {{11, 10}, ACT_MOVE_LEFT},
  {{11,  9}, ACT_STOP_UP},
  {{11,  9}, ACT_MOVE_LEFT},
  {{11,  8}, ACT_STOP_UP},
  {{11,  8}, ACT_MOVE_LEFT},
  {{11,  7}, ACT_STOP_UP},
  {{11,  7}, ACT_MOVE_LEFT},
  {{11,  6}, ACT_MOVE_UP},
  {{10,  6}, ACT_MOVE_UP},
  {{ 9,  6}, ACT_STOP_RIGHT},
  {{ 9,  6}, ACT_MOVE_UP},
  {{ 8,  6}, ACT_STOP_RIGHT},
  {{ 8,  6}, ACT_MOVE_UP},
  {{ 7,  6}, ACT_MOVE_RIGHT},
  {{ 7,  7}, ACT_MOVE_RIGHT},
  {{ 7,  8}, ACT_STOP_LEFT},
  {{ 7,  8}, ACT_STOP_UP},
  {{ 7,  8}, ACT_MOVE_RIGHT},
  {{ 7,  9}, ACT_MOVE_RIGHT},
  {{ 7, 10}, ACT_MOVE_RIGHT},
  {{ 7, 11}, ACT_STOP_LEFT},
  {{ 7, 11}, ACT_MOVE_DOWN},
  {{ 8, 11}, ACT_STOP_LEFT},
  {{ 8, 11}, ACT_MOVE_DOWN},
  {{ 9, 11}, ACT_STOP_LEFT},
  {{ 9, 11}, ACT_MOVE_DOWN},
  {{10, 11}, ACT_MOVE_RIGHT},
  {{10, 12}, ACT_MOVE_RIGHT},
  {{10, 13}, ACT_MOVE_RIGHT},
  {{10, 14}, ACT_STOP_LEFT},
  {{10, 14}, ACT_MOVE_UP},
  {{ 9, 14}, ACT_STOP_LEFT},
  {{ 9, 14}, ACT_MOVE_UP},
  {{ 8, 14}, ACT_STOP_LEFT},
  {{ 8, 14}, ACT_MOVE_UP},
  {{ 7, 14}, ACT_STOP_LEFT},
  {{ 7, 14}, ACT_MOVE_RIGHT},
  {{ 7, 15}, ACT_MOVE_RIGHT},
  {{ 7, 16}, ACT_MOVE_RIGHT},
  {{ 7, 17}, ACT_STOP_LEFT},
  {{ 7, 17}, ACT_MOVE_DOWN},
  {{ 8, 17}, ACT_STOP_LEFT},
  {{ 8, 17}, ACT_MOVE_DOWN},
  {{ 9, 17}, ACT_STOP_LEFT},
  {{ 9, 17}, ACT_MOVE_DOWN},
  {{10, 17}, ACT_STOP_LEFT},
  {{10, 17}, ACT_MOVE_RIGHT},
  {{10, 18}, ACT_STOP_RIGHT},
  {{10, 18}, ACT_MOVE_UP},
  {{ 9, 18}, ACT_STOP_RIGHT},
  {{ 9, 18}, ACT_MOVE_UP},
  {{ 8, 18}, ACT_STOP_RIGHT},
  {{ 8, 18}, ACT_MOVE_UP},
  {{ 7, 18}, ACT_STOP_RIGHT},
  {{ 7, 18}, ACT_MOVE_UP},
  {{ 6, 18}, ACT_MOVE_RIGHT},
  {{ 6, 19}, ACT_MOVE_RIGHT},
  {{ 6, 20}, ACT_STOP_DOWN},
  {{ 6, 20}, ACT_MOVE_RIGHT},
  {{ 6, 21}, ACT_STOP_DOWN},
  {{ 6, 21}, ACT_MOVE_RIGHT},
  {{ 6, 22}, ACT_MOVE_DOWN},
  {{ 7, 22}, ACT_MOVE_DOWN},
  {{ 8, 22}, ACT_STOP_UP},
  {{ 8, 22}, ACT_STOP_RIGHT},
  {{ 8, 22}, ACT_MOVE_DOWN},
  {{ 9, 22}, ACT_MOVE_DOWN},
  {{10, 22}, ACT_MOVE_DOWN},
  {{11, 22}, ACT_STOP_UP},
  {{11, 22}, ACT_MOVE_LEFT},
  {{11, 21}, ACT_STOP_UP},
  {{11, 21}, ACT_MOVE_LEFT},
  {{11, 20}, ACT_STOP_UP},
  {{11, 20}, ACT_MOVE_LEFT},
  {{11, 19}, ACT_MOVE_DOWN},
  {{12, 19}, ACT_MOVE_DOWN},
  {{13, 19}, ACT_MOVE_DOWN},
  {{14, 19}, ACT_STOP_UP},
  {{14, 19}, ACT_MOVE_RIGHT},
  {{14, 20}, ACT_STOP_UP},
  {{14, 20}, ACT_MOVE_RIGHT},
  {{14, 21}, ACT_STOP_UP},
  {{14, 21}, ACT_MOVE_RIGHT},
  {{14, 22}, ACT_STOP_UP},
  {{14, 22}, ACT_MOVE_DOWN},
  {{15, 22}, ACT_MOVE_DOWN},
  {{16, 22}, ACT_MOVE_DOWN},
  {{17, 22}, ACT_STOP_UP},
  {{17, 22}, ACT_MOVE_LEFT},
  {{17, 21}, ACT_STOP_UP},
  {{17, 21}, ACT_MOVE_LEFT},
  {{17, 20}, ACT_STOP_UP},
  {{17, 20}, ACT_MOVE_LEFT},
  {{17, 19}, ACT_STOP_UP},
  {{17, 19}, ACT_MOVE_DOWN},
  {{18, 19}, ACT_STOP_DOWN},
  {{18, 19}, ACT_MOVE_RIGHT},
  {{18, 20}, ACT_STOP_DOWN},
  {{18, 20}, ACT_MOVE_RIGHT},
  {{18, 21}, ACT_STOP_DOWN},
  {{18, 21}, ACT_MOVE_RIGHT},
  {{18, 22}, ACT_STOP_DOWN},
  {{18, 22}, ACT_MOVE_RIGHT},
  {{18, 23}, ACT_MOVE_DOWN},
  {{19, 23}, ACT_MOVE_DOWN},
  {{20, 23}, ACT_STOP_LEFT},
  {{20, 23}, ACT_MOVE_DOWN},
  {{21, 23}, ACT_STOP_LEFT},
  {{21, 23}, ACT_MOVE_DOWN},
  {{22, 23}, ACT_MOVE_LEFT},
  {{22, 22}, ACT_MOVE_LEFT},
  {{22, 21}, ACT_STOP_RIGHT},
  {{22, 21}, ACT_STOP_DOWN},
  {{22, 21}, ACT_MOVE_LEFT},
  {{22, 20}, ACT_MOVE_LEFT},
  {{22, 19}, ACT_MOVE_LEFT},
  {{22, 18}, ACT_STOP_RIGHT},
  {{22, 18}, ACT_MOVE_UP},
  {{21, 18}, ACT_STOP_RIGHT},
  {{21, 18}, ACT_MOVE_UP},
  {{20, 18}, ACT_STOP_RIGHT},
  {{20, 18}, ACT_MOVE_UP},
  {{19, 18}, ACT_MOVE_LEFT},
  {{19, 17}, ACT_MOVE_LEFT},
  {{19, 16}, ACT_MOVE_LEFT},
  {{19, 15}, ACT_STOP_RIGHT},
  {{19, 15}, ACT_MOVE_DOWN},
  {{20, 15}, ACT_STOP_RIGHT},
  {{20, 15}, ACT_MOVE_DOWN},
  {{21, 15}, ACT_STOP_RIGHT},
  {{21, 15}, ACT_MOVE_DOWN},
  {{22, 15}, ACT_STOP_RIGHT},
  {{22, 15}, ACT_MOVE_LEFT},
  {{22, 14}, ACT_MOVE_LEFT},
  {{22, 13}, ACT_MOVE_LEFT},
  {{22, 12}, ACT_STOP_RIGHT},
  {{22, 12}, ACT_MOVE_UP},
  {{21, 12}, ACT_STOP_RIGHT},
  {{21, 12}, ACT_MOVE_UP},
  {{20, 12}, ACT_STOP_RIGHT},
  {{20, 12}, ACT_MOVE_UP},
  {{19, 12}, ACT_STOP_RIGHT},
  {{19, 12}, ACT_MOVE_LEFT},
  {{19, 11}, ACT_STOP_LEFT},
  {{19, 11}, ACT_MOVE_DOWN},
  {{20, 11}, ACT_STOP_LEFT},
  {{20, 11}, ACT_MOVE_DOWN},
  {{21, 11}, ACT_STOP_LEFT},
  {{21, 11}, ACT_MOVE_DOWN},
  {{22, 11}, ACT_STOP_LEFT},
  {{22, 11}, ACT_MOVE_DOWN},
  {{23, 11}, ACT_MOVE_LEFT},
  {{23, 10}, ACT_MOVE_LEFT},
  {{23,  9}, ACT_STOP_UP},
  {{23,  9}, ACT_MOVE_LEFT},
  {{23,  8}, ACT_STOP_UP},
  {{23,  8}, ACT_MOVE_LEFT},
  {{23,  7}, ACT_MOVE_UP},
  {{22,  7}, ACT_MOVE_UP},
  {{21,  7}, ACT_STOP_DOWN},
  {{21,  7}, ACT_STOP_LEFT},
  {{21,  7}, ACT_MOVE_UP},
  {{20,  7}, ACT_MOVE_UP},
  {{19,  7}, ACT_MOVE_UP},
  {{18,  7}, ACT_STOP_DOWN},
  {{18,  7}, ACT_MOVE_RIGHT},
  {{18,  8}, ACT_STOP_DOWN},
  {{18,  8}, ACT_MOVE_RIGHT},
  {{18,  9}, ACT_STOP_DOWN},
  {{18,  9}, ACT_MOVE_RIGHT},
  {{18, 10}, ACT_MOVE_UP},
  {{17, 10}, ACT_MOVE_UP},
  {{16, 10}, ACT_MOVE_UP},
  {{15, 10}, ACT_STOP_DOWN},
  {{15, 10}, ACT_MOVE_LEFT},
  {{15,  9}, ACT_STOP_DOWN},
  {{15,  9}, ACT_MOVE_LEFT},
  {{15,  8}, ACT_STOP_DOWN},
  {{15,  8}, ACT_MOVE_LEFT},
  {{15,  7}, ACT_STOP_DOWN},
  {{15,  7}, ACT_MOVE_UP},
  {{14,  7}, ACT_MOVE_UP},
  {{13,  7}, ACT_MOVE_UP},
  {{12,  7}, ACT_STOP_DOWN},
  {{12,  7}, ACT_MOVE_RIGHT},
  {{12,  8}, ACT_STOP_DOWN},
  {{12,  8}, ACT_MOVE_RIGHT},
  {{12,  9}, ACT_STOP_DOWN},
  {{12,  9}, ACT_MOVE_RIGHT},
  {{12, 10}, ACT_STOP_DOWN},
  {{12, 10}, ACT_MOVE_UP},
};
vector<pair<P, char>> build_sequence_center_few_dogs = {
  {{11, 10}, ACT_STOP_UP},
  {{11, 10}, ACT_MOVE_LEFT},
  {{11,  9}, ACT_STOP_UP},
  {{11,  9}, ACT_MOVE_LEFT},
  {{11,  8}, ACT_STOP_UP},
  {{11,  8}, ACT_MOVE_LEFT},
  {{11,  7}, ACT_STOP_UP},
  {{11,  7}, ACT_MOVE_LEFT},
  {{11,  6}, ACT_MOVE_UP},
  {{10,  6}, ACT_MOVE_UP},
  {{ 9,  6}, ACT_STOP_RIGHT},
  {{ 9,  6}, ACT_MOVE_UP},
  {{ 8,  6}, ACT_STOP_RIGHT},
  {{ 8,  6}, ACT_MOVE_UP},
  {{ 7,  6}, ACT_MOVE_RIGHT},
  {{ 7,  7}, ACT_MOVE_RIGHT},
  {{ 7,  8}, ACT_STOP_LEFT},
  {{ 7,  8}, ACT_STOP_UP},
  {{ 7,  8}, ACT_MOVE_RIGHT},
  {{ 7,  9}, ACT_MOVE_RIGHT},
  {{ 7, 10}, ACT_MOVE_RIGHT},
  {{ 7, 11}, ACT_STOP_LEFT},
  {{ 7, 11}, ACT_MOVE_DOWN},
  {{ 8, 11}, ACT_STOP_LEFT},
  {{ 8, 11}, ACT_MOVE_DOWN},
  {{ 9, 11}, ACT_STOP_LEFT},
  {{ 9, 11}, ACT_MOVE_DOWN},
  {{10, 11}, ACT_MOVE_RIGHT},
  {{10, 12}, ACT_MOVE_DOWN},
  {{11, 12}, ACT_MOVE_RIGHT},
  {{11, 13}, ACT_MOVE_RIGHT},
  {{11, 14}, ACT_STOP_LEFT},
  {{11, 14}, ACT_MOVE_UP},
  {{10, 14}, ACT_STOP_LEFT},
  {{10, 14}, ACT_MOVE_UP},
  {{ 9, 14}, ACT_STOP_LEFT},
  {{ 9, 14}, ACT_MOVE_UP},
  {{ 8, 14}, ACT_STOP_LEFT},
  {{ 8, 14}, ACT_MOVE_UP},
  {{ 7, 14}, ACT_STOP_LEFT},
  {{ 7, 14}, ACT_MOVE_RIGHT},
  {{ 7, 15}, ACT_STOP_RIGHT},
  {{ 7, 15}, ACT_MOVE_DOWN},
  {{ 8, 15}, ACT_STOP_RIGHT},
  {{ 8, 15}, ACT_MOVE_DOWN},
  {{ 9, 15}, ACT_STOP_RIGHT},
  {{ 9, 15}, ACT_MOVE_DOWN},
  {{10, 15}, ACT_STOP_RIGHT},
  {{10, 15}, ACT_MOVE_DOWN},
  {{11, 15}, ACT_MOVE_RIGHT},
  {{11, 16}, ACT_MOVE_RIGHT},
  {{11, 17}, ACT_STOP_LEFT},
  {{11, 17}, ACT_STOP_RIGHT},
  {{11, 17}, ACT_MOVE_UP},
  {{10, 17}, ACT_MOVE_RIGHT},
  {{10, 18}, ACT_STOP_RIGHT},
  {{10, 18}, ACT_MOVE_UP},
  {{ 9, 18}, ACT_STOP_RIGHT},
  {{ 9, 18}, ACT_MOVE_UP},
  {{ 8, 18}, ACT_STOP_RIGHT},
  {{ 8, 18}, ACT_MOVE_UP},
  {{ 7, 18}, ACT_STOP_RIGHT},
  {{ 7, 18}, ACT_MOVE_UP},
  {{ 6, 18}, ACT_MOVE_RIGHT},
  {{ 6, 19}, ACT_MOVE_RIGHT},
  {{ 6, 20}, ACT_STOP_DOWN},
  {{ 6, 20}, ACT_MOVE_RIGHT},
  {{ 6, 21}, ACT_STOP_DOWN},
  {{ 6, 21}, ACT_MOVE_RIGHT},
  {{ 6, 22}, ACT_MOVE_DOWN},
  {{ 7, 22}, ACT_MOVE_DOWN},
  {{ 8, 22}, ACT_STOP_UP},
  {{ 8, 22}, ACT_STOP_RIGHT},
  {{ 8, 22}, ACT_MOVE_DOWN},
  {{ 9, 22}, ACT_MOVE_DOWN},
  {{10, 22}, ACT_MOVE_DOWN},
  {{11, 22}, ACT_STOP_UP},
  {{11, 22}, ACT_MOVE_LEFT},
  {{11, 21}, ACT_STOP_UP},
  {{11, 21}, ACT_MOVE_LEFT},
  {{11, 20}, ACT_STOP_UP},
  {{11, 20}, ACT_MOVE_LEFT},
  {{11, 19}, ACT_MOVE_DOWN},
  {{12, 19}, ACT_MOVE_LEFT},
  {{12, 18}, ACT_MOVE_DOWN},
  {{13, 18}, ACT_MOVE_DOWN},
  {{14, 18}, ACT_STOP_UP},
  {{14, 18}, ACT_MOVE_RIGHT},
  {{14, 19}, ACT_STOP_UP},
  {{14, 19}, ACT_MOVE_RIGHT},
  {{14, 20}, ACT_STOP_UP},
  {{14, 20}, ACT_MOVE_RIGHT},
  {{14, 21}, ACT_STOP_UP},
  {{14, 21}, ACT_MOVE_RIGHT},
  {{14, 22}, ACT_STOP_UP},
  {{14, 22}, ACT_MOVE_DOWN},
  {{15, 22}, ACT_STOP_DOWN},
  {{15, 22}, ACT_MOVE_LEFT},
  {{15, 21}, ACT_STOP_DOWN},
  {{15, 21}, ACT_MOVE_LEFT},
  {{15, 20}, ACT_STOP_DOWN},
  {{15, 20}, ACT_MOVE_LEFT},
  {{15, 19}, ACT_STOP_DOWN},
  {{15, 19}, ACT_MOVE_LEFT},
  {{15, 18}, ACT_MOVE_DOWN},
  {{16, 18}, ACT_MOVE_DOWN},
  {{17, 18}, ACT_STOP_UP},
  {{17, 18}, ACT_STOP_DOWN},
  {{17, 18}, ACT_MOVE_RIGHT},
  {{17, 19}, ACT_MOVE_DOWN},
  {{18, 19}, ACT_STOP_DOWN},
  {{18, 19}, ACT_MOVE_RIGHT},
  {{18, 20}, ACT_STOP_DOWN},
  {{18, 20}, ACT_MOVE_RIGHT},
  {{18, 21}, ACT_STOP_DOWN},
  {{18, 21}, ACT_MOVE_RIGHT},
  {{18, 22}, ACT_STOP_DOWN},
  {{18, 22}, ACT_MOVE_RIGHT},
  {{18, 23}, ACT_MOVE_DOWN},
  {{19, 23}, ACT_MOVE_DOWN},
  {{20, 23}, ACT_STOP_LEFT},
  {{20, 23}, ACT_MOVE_DOWN},
  {{21, 23}, ACT_STOP_LEFT},
  {{21, 23}, ACT_MOVE_DOWN},
  {{22, 23}, ACT_MOVE_LEFT},
  {{22, 22}, ACT_MOVE_LEFT},
  {{22, 21}, ACT_STOP_RIGHT},
  {{22, 21}, ACT_STOP_DOWN},
  {{22, 21}, ACT_MOVE_LEFT},
  {{22, 20}, ACT_MOVE_LEFT},
  {{22, 19}, ACT_MOVE_LEFT},
  {{22, 18}, ACT_STOP_RIGHT},
  {{22, 18}, ACT_MOVE_UP},
  {{21, 18}, ACT_STOP_RIGHT},
  {{21, 18}, ACT_MOVE_UP},
  {{20, 18}, ACT_STOP_RIGHT},
  {{20, 18}, ACT_MOVE_UP},
  {{19, 18}, ACT_MOVE_LEFT},
  {{19, 17}, ACT_MOVE_UP},
  {{18, 17}, ACT_MOVE_LEFT},
  {{18, 16}, ACT_MOVE_LEFT},
  {{18, 15}, ACT_STOP_RIGHT},
  {{18, 15}, ACT_MOVE_DOWN},
  {{19, 15}, ACT_STOP_RIGHT},
  {{19, 15}, ACT_MOVE_DOWN},
  {{20, 15}, ACT_STOP_RIGHT},
  {{20, 15}, ACT_MOVE_DOWN},
  {{21, 15}, ACT_STOP_RIGHT},
  {{21, 15}, ACT_MOVE_DOWN},
  {{22, 15}, ACT_STOP_RIGHT},
  {{22, 15}, ACT_MOVE_LEFT},
  {{22, 14}, ACT_STOP_LEFT},
  {{22, 14}, ACT_MOVE_UP},
  {{21, 14}, ACT_STOP_LEFT},
  {{21, 14}, ACT_MOVE_UP},
  {{20, 14}, ACT_STOP_LEFT},
  {{20, 14}, ACT_MOVE_UP},
  {{19, 14}, ACT_STOP_LEFT},
  {{19, 14}, ACT_MOVE_UP},
  {{18, 14}, ACT_MOVE_LEFT},
  {{18, 13}, ACT_MOVE_LEFT},
  {{18, 12}, ACT_STOP_RIGHT},
  {{18, 12}, ACT_STOP_LEFT},
  {{18, 12}, ACT_MOVE_DOWN},
  {{19, 12}, ACT_MOVE_LEFT},
  {{19, 11}, ACT_STOP_LEFT},
  {{19, 11}, ACT_MOVE_DOWN},
  {{20, 11}, ACT_STOP_LEFT},
  {{20, 11}, ACT_MOVE_DOWN},
  {{21, 11}, ACT_STOP_LEFT},
  {{21, 11}, ACT_MOVE_DOWN},
  {{22, 11}, ACT_STOP_LEFT},
  {{22, 11}, ACT_MOVE_DOWN},
  {{23, 11}, ACT_MOVE_LEFT},
  {{23, 10}, ACT_MOVE_LEFT},
  {{23,  9}, ACT_STOP_UP},
  {{23,  9}, ACT_MOVE_LEFT},
  {{23,  8}, ACT_STOP_UP},
  {{23,  8}, ACT_MOVE_LEFT},
  {{23,  7}, ACT_MOVE_UP},
  {{22,  7}, ACT_MOVE_UP},
  {{21,  7}, ACT_STOP_DOWN},
  {{21,  7}, ACT_STOP_LEFT},
  {{21,  7}, ACT_MOVE_UP},
  {{20,  7}, ACT_MOVE_UP},
  {{19,  7}, ACT_MOVE_UP},
  {{18,  7}, ACT_STOP_DOWN},
  {{18,  7}, ACT_MOVE_RIGHT},
  {{18,  8}, ACT_STOP_DOWN},
  {{18,  8}, ACT_MOVE_RIGHT},
  {{18,  9}, ACT_STOP_DOWN},
  {{18,  9}, ACT_MOVE_RIGHT},
  {{18, 10}, ACT_MOVE_UP},
  {{17, 10}, ACT_MOVE_RIGHT},
  {{17, 11}, ACT_MOVE_UP},
  {{16, 11}, ACT_MOVE_UP},
  {{15, 11}, ACT_STOP_DOWN},
  {{15, 11}, ACT_MOVE_LEFT},
  {{15, 10}, ACT_STOP_DOWN},
  {{15, 10}, ACT_MOVE_LEFT},
  {{15,  9}, ACT_STOP_DOWN},
  {{15,  9}, ACT_MOVE_LEFT},
  {{15,  8}, ACT_STOP_DOWN},
  {{15,  8}, ACT_MOVE_LEFT},
  {{15,  7}, ACT_STOP_DOWN},
  {{15,  7}, ACT_MOVE_UP},
  {{14,  7}, ACT_STOP_UP},
  {{14,  7}, ACT_MOVE_RIGHT},
  {{14,  8}, ACT_STOP_UP},
  {{14,  8}, ACT_MOVE_RIGHT},
  {{14,  9}, ACT_STOP_UP},
  {{14,  9}, ACT_MOVE_RIGHT},
  {{14, 10}, ACT_STOP_UP},
  {{14, 10}, ACT_MOVE_RIGHT},
  {{14, 11}, ACT_MOVE_UP},
  {{13, 11}, ACT_MOVE_UP},
  {{12, 11}, ACT_STOP_DOWN},
  {{12, 11}, ACT_STOP_UP},
  {{12, 11}, ACT_MOVE_LEFT},
  {{12, 10}, ACT_MOVE_UP},
};
vector<pair<P, char>> build_sequence_center_nodog = {
  {{11, 10}, ACT_STOP_UP},
  {{11, 10}, ACT_MOVE_LEFT},
  {{11,  9}, ACT_STOP_UP},
  {{11,  9}, ACT_MOVE_LEFT},
  {{11,  8}, ACT_STOP_UP},
  {{11,  8}, ACT_MOVE_LEFT},
  {{11,  7}, ACT_STOP_UP},
  {{11,  7}, ACT_MOVE_LEFT},
  {{11,  6}, ACT_MOVE_UP},
  {{10,  6}, ACT_MOVE_UP},
  {{ 9,  6}, ACT_STOP_RIGHT},
  {{ 9,  6}, ACT_MOVE_UP},
  {{ 8,  6}, ACT_STOP_RIGHT},
  {{ 8,  6}, ACT_MOVE_UP},
  {{ 7,  6}, ACT_MOVE_RIGHT},
  {{ 7,  7}, ACT_MOVE_RIGHT},
  {{ 7,  8}, ACT_STOP_LEFT},
  {{ 7,  8}, ACT_STOP_UP},
  {{ 7,  8}, ACT_MOVE_RIGHT},
  {{ 7,  9}, ACT_MOVE_RIGHT},
  {{ 7, 10}, ACT_MOVE_RIGHT},
  {{ 7, 11}, ACT_MOVE_RIGHT},
  {{ 7, 12}, ACT_STOP_UP},
  {{ 7, 12}, ACT_STOP_LEFT},
  {{ 7, 12}, ACT_MOVE_DOWN},
  {{ 8, 12}, ACT_STOP_LEFT},
  {{ 8, 12}, ACT_MOVE_DOWN},
  {{ 9, 12}, ACT_STOP_LEFT},
  {{ 9, 12}, ACT_MOVE_DOWN},
  {{10, 12}, ACT_MOVE_DOWN},
  {{11, 12}, ACT_STOP_LEFT},
  {{11, 12}, ACT_MOVE_DOWN},
  {{12, 12}, ACT_STOP_LEFT},
  {{12, 12}, ACT_MOVE_DOWN},
  {{13, 12}, ACT_STOP_LEFT},
  {{13, 12}, ACT_STOP_DOWN},
  {{13, 12}, ACT_MOVE_RIGHT},
  {{13, 13}, ACT_STOP_DOWN},
  {{13, 13}, ACT_MOVE_RIGHT},
  {{13, 14}, ACT_STOP_DOWN},
  {{13, 14}, ACT_MOVE_UP},
  {{12, 14}, ACT_MOVE_UP},
  {{11, 14}, ACT_MOVE_UP},
  {{10, 14}, ACT_STOP_RIGHT},
  {{10, 14}, ACT_MOVE_UP},
  {{ 9, 14}, ACT_STOP_RIGHT},
  {{ 9, 14}, ACT_MOVE_UP},
  {{ 8, 14}, ACT_STOP_RIGHT},
  {{ 8, 14}, ACT_MOVE_UP},
  {{ 7, 14}, ACT_MOVE_RIGHT},
  {{ 7, 15}, ACT_MOVE_RIGHT},
  {{ 7, 16}, ACT_STOP_LEFT},
  {{ 7, 16}, ACT_STOP_UP},
  {{ 7, 16}, ACT_MOVE_DOWN},
  {{ 8, 16}, ACT_MOVE_DOWN},
  {{ 9, 16}, ACT_MOVE_DOWN},
  {{10, 16}, ACT_MOVE_RIGHT},
  {{10, 17}, ACT_MOVE_RIGHT},
  {{10, 18}, ACT_STOP_RIGHT},
  {{10, 18}, ACT_MOVE_UP},
  {{ 9, 18}, ACT_STOP_RIGHT},
  {{ 9, 18}, ACT_MOVE_UP},
  {{ 8, 18}, ACT_STOP_RIGHT},
  {{ 8, 18}, ACT_MOVE_UP},
  {{ 7, 18}, ACT_STOP_RIGHT},
  {{ 7, 18}, ACT_MOVE_UP},
  {{ 6, 18}, ACT_MOVE_RIGHT},
  {{ 6, 19}, ACT_MOVE_RIGHT},
  {{ 6, 20}, ACT_STOP_DOWN},
  {{ 6, 20}, ACT_MOVE_RIGHT},
  {{ 6, 21}, ACT_STOP_DOWN},
  {{ 6, 21}, ACT_MOVE_RIGHT},
  {{ 6, 22}, ACT_MOVE_DOWN},
  {{ 7, 22}, ACT_MOVE_DOWN},
  {{ 8, 22}, ACT_STOP_UP},
  {{ 8, 22}, ACT_STOP_RIGHT},
  {{ 8, 22}, ACT_MOVE_DOWN},
  {{ 9, 22}, ACT_MOVE_DOWN},
  {{10, 22}, ACT_MOVE_DOWN},
  {{11, 22}, ACT_MOVE_DOWN},
  {{12, 22}, ACT_STOP_UP},
  {{12, 22}, ACT_STOP_RIGHT},
  {{12, 22}, ACT_MOVE_LEFT},
  {{12, 21}, ACT_STOP_UP},
  {{12, 21}, ACT_MOVE_LEFT},
  {{12, 20}, ACT_STOP_UP},
  {{12, 20}, ACT_MOVE_LEFT},
  {{12, 19}, ACT_MOVE_LEFT},
  {{12, 18}, ACT_STOP_UP},
  {{12, 18}, ACT_MOVE_LEFT},
  {{12, 17}, ACT_STOP_UP},
  {{12, 17}, ACT_MOVE_LEFT},
  {{12, 16}, ACT_STOP_UP},
  {{12, 16}, ACT_STOP_LEFT},
  {{12, 16}, ACT_MOVE_DOWN},
  {{13, 16}, ACT_STOP_LEFT},
  {{13, 16}, ACT_MOVE_DOWN},
  {{14, 16}, ACT_MOVE_RIGHT},
  {{14, 17}, ACT_MOVE_RIGHT},
  {{14, 18}, ACT_MOVE_RIGHT},
  {{14, 19}, ACT_STOP_DOWN},
  {{14, 19}, ACT_MOVE_RIGHT},
  {{14, 20}, ACT_STOP_DOWN},
  {{14, 20}, ACT_MOVE_RIGHT},
  {{14, 21}, ACT_STOP_DOWN},
  {{14, 21}, ACT_MOVE_RIGHT},
  {{14, 22}, ACT_MOVE_DOWN},
  {{15, 22}, ACT_MOVE_DOWN},
  {{16, 22}, ACT_STOP_UP},
  {{16, 22}, ACT_STOP_RIGHT},
  {{16, 22}, ACT_MOVE_LEFT},
  {{16, 21}, ACT_MOVE_LEFT},
  {{16, 20}, ACT_MOVE_LEFT},
  {{16, 19}, ACT_MOVE_DOWN},
  {{17, 19}, ACT_MOVE_DOWN},
  {{18, 19}, ACT_STOP_DOWN},
  {{18, 19}, ACT_MOVE_RIGHT},
  {{18, 20}, ACT_STOP_DOWN},
  {{18, 20}, ACT_MOVE_RIGHT},
  {{18, 21}, ACT_STOP_DOWN},
  {{18, 21}, ACT_MOVE_RIGHT},
  {{18, 22}, ACT_STOP_DOWN},
  {{18, 22}, ACT_MOVE_RIGHT},
  {{18, 23}, ACT_MOVE_DOWN},
  {{19, 23}, ACT_MOVE_DOWN},
  {{20, 23}, ACT_STOP_LEFT},
  {{20, 23}, ACT_MOVE_DOWN},
  {{21, 23}, ACT_STOP_LEFT},
  {{21, 23}, ACT_MOVE_DOWN},
  {{22, 23}, ACT_MOVE_LEFT},
  {{22, 22}, ACT_MOVE_LEFT},
  {{22, 21}, ACT_STOP_RIGHT},
  {{22, 21}, ACT_STOP_DOWN},
  {{22, 21}, ACT_MOVE_LEFT},
  {{22, 20}, ACT_MOVE_LEFT},
  {{22, 19}, ACT_MOVE_LEFT},
  {{22, 18}, ACT_MOVE_LEFT},
  {{22, 17}, ACT_STOP_RIGHT},
  {{22, 17}, ACT_STOP_DOWN},
  {{22, 17}, ACT_MOVE_UP},
  {{21, 17}, ACT_STOP_RIGHT},
  {{21, 17}, ACT_MOVE_UP},
  {{20, 17}, ACT_STOP_RIGHT},
  {{20, 17}, ACT_MOVE_UP},
  {{19, 17}, ACT_MOVE_UP},
  {{18, 17}, ACT_STOP_RIGHT},
  {{18, 17}, ACT_MOVE_UP},
  {{17, 17}, ACT_STOP_RIGHT},
  {{17, 17}, ACT_MOVE_UP},
  {{16, 17}, ACT_STOP_RIGHT},
  {{16, 17}, ACT_STOP_UP},
  {{16, 17}, ACT_MOVE_LEFT},
  {{16, 16}, ACT_STOP_UP},
  {{16, 16}, ACT_MOVE_LEFT},
  {{16, 15}, ACT_STOP_UP},
  {{16, 15}, ACT_MOVE_DOWN},
  {{17, 15}, ACT_MOVE_DOWN},
  {{18, 15}, ACT_MOVE_DOWN},
  {{19, 15}, ACT_STOP_LEFT},
  {{19, 15}, ACT_MOVE_DOWN},
  {{20, 15}, ACT_STOP_LEFT},
  {{20, 15}, ACT_MOVE_DOWN},
  {{21, 15}, ACT_STOP_LEFT},
  {{21, 15}, ACT_MOVE_DOWN},
  {{22, 15}, ACT_MOVE_LEFT},
  {{22, 14}, ACT_MOVE_LEFT},
  {{22, 13}, ACT_STOP_RIGHT},
  {{22, 13}, ACT_STOP_DOWN},
  {{22, 13}, ACT_MOVE_UP},
  {{21, 13}, ACT_MOVE_UP},
  {{20, 13}, ACT_MOVE_UP},
  {{19, 13}, ACT_MOVE_LEFT},
  {{19, 12}, ACT_MOVE_LEFT},
  {{19, 11}, ACT_STOP_LEFT},
  {{19, 11}, ACT_MOVE_DOWN},
  {{20, 11}, ACT_STOP_LEFT},
  {{20, 11}, ACT_MOVE_DOWN},
  {{21, 11}, ACT_STOP_LEFT},
  {{21, 11}, ACT_MOVE_DOWN},
  {{22, 11}, ACT_STOP_LEFT},
  {{22, 11}, ACT_MOVE_DOWN},
  {{23, 11}, ACT_MOVE_LEFT},
  {{23, 10}, ACT_MOVE_LEFT},
  {{23,  9}, ACT_STOP_UP},
  {{23,  9}, ACT_MOVE_LEFT},
  {{23,  8}, ACT_STOP_UP},
  {{23,  8}, ACT_MOVE_LEFT},
  {{23,  7}, ACT_MOVE_UP},
  {{22,  7}, ACT_MOVE_UP},
  {{21,  7}, ACT_STOP_DOWN},
  {{21,  7}, ACT_STOP_LEFT},
  {{21,  7}, ACT_MOVE_UP},
  {{20,  7}, ACT_MOVE_UP},
  {{19,  7}, ACT_MOVE_UP},
  {{18,  7}, ACT_MOVE_UP},
  {{17,  7}, ACT_STOP_DOWN},
  {{17,  7}, ACT_STOP_LEFT},
  {{17,  7}, ACT_MOVE_RIGHT},
  {{17,  8}, ACT_STOP_DOWN},
  {{17,  8}, ACT_MOVE_RIGHT},
  {{17,  9}, ACT_STOP_DOWN},
  {{17,  9}, ACT_MOVE_RIGHT},
  {{17, 10}, ACT_MOVE_RIGHT},
  {{17, 11}, ACT_STOP_DOWN},
  {{17, 11}, ACT_MOVE_RIGHT},
  {{17, 12}, ACT_STOP_DOWN},
  {{17, 12}, ACT_MOVE_RIGHT},
  {{17, 13}, ACT_STOP_DOWN},
  {{17, 13}, ACT_STOP_RIGHT},
  {{17, 13}, ACT_MOVE_UP},
  {{16, 13}, ACT_STOP_RIGHT},
  {{16, 13}, ACT_MOVE_UP},
  {{15, 13}, ACT_MOVE_LEFT},
  {{15, 12}, ACT_MOVE_LEFT},
  {{15, 11}, ACT_MOVE_LEFT},
  {{15, 10}, ACT_STOP_UP},
  {{15, 10}, ACT_MOVE_LEFT},
  {{15,  9}, ACT_STOP_UP},
  {{15,  9}, ACT_MOVE_LEFT},
  {{15,  8}, ACT_STOP_UP},
  {{15,  8}, ACT_MOVE_LEFT},
  {{15,  7}, ACT_MOVE_UP},
  {{14,  7}, ACT_MOVE_UP},
  {{13,  7}, ACT_STOP_DOWN},
  {{13,  7}, ACT_STOP_LEFT},
  {{13,  7}, ACT_MOVE_RIGHT},
  {{13,  8}, ACT_MOVE_RIGHT},
  {{13,  9}, ACT_MOVE_RIGHT},
  {{13, 10}, ACT_MOVE_UP},
  {{12, 10}, ACT_MOVE_UP},
};

vector<P> phase_1_startpos = { {19, 17}, {10, 12}, {12, 19}, {17, 10} };
vector<P> phase_1_waitpos = { {18, 12}, {11, 17}, {17, 18}, {12, 11} };
vector<P> phase_1_startpos_few_dogs = { {18, 16}, {11, 13}, {13, 18}, {16, 11} };
vector<P> phase_1_waitpos_few_dogs = { {17, 13}, {12, 16}, {16, 17}, {13, 12} };

vector<P> final_operation_pos;

// マスの情報
const int NONE = 0;
const int PASSAGE = 1;
const int BLOCK_ACTION_POINT = 2;
const int BLOCK_ACTION_POINT_SPE = 3;
const int BLOCK_POINT_SPE = 4;
const int BLOCK_POINT = 5;
const int BLOCK = 6;
const int BLOCK_FOR_DOG = 10;
const int DOG_CATCH_ZONE = 7;
const int PRE_CATCH_ZONE = 8;
const int CATCH_ZONE = 9;

// ペットの種類
const int COW = 0;
const int PIG = 1;
const int RAB = 2;
const int DOG = 3;
const int CAT = 4;

const int FEW_DOGS_THRESHOLD = 3;

// ペットの情報
int N;                      // 数
vector<P> pp(30);           // 位置
vector<int> pt(30);         // 種類
vector<bool> ps(30, false); // 捕獲済みか否か
vector<vector<int>> pf(H, vector<int>(W, 0)); // ペットの位置
int pet_remains;            // 未捕獲の動物の数
bool dog_exists = false;    // 犬が存在するかどうか
vector<P> pnp(30);          // ペットから見た最も近い通路マス

// 人の情報
int M;                // 数
vector<P> hp(30);     // 位置
// 人が抱えている仕事
struct HumanS {
  int job;        // 仕事
                  // 0: 初期状態、または全ての動物の捕獲が完了し、何もしない状態
                  // 1: フェーズ1の地形作成開始ポイントへの移動
                  // 2: フェーズ1の地形作成
                  // 3: フェーズ1の待機ポイントへ移動、待機
                  // 4:
                  // 5:
                  // 6:
                  // 7:
  int meta;       // 補助的な情報を入れる
  int target;     // 捕獲対象の動物ID
  P subject_pos;  // 目的地
};
vector<HumanS> human(30);
vector<vector<int>> hf(H, vector<int>(W, 0)); // 人の位置

// 捕獲ゾーン管理
vector<vector<int>> catch_zone_id(H, vector<int>(W, -1));
vector<P> catch_zone_pos;       // 捕獲ゾーン基準位置
vector<int> catch_zone_status;  // 捕獲ゾーンの状態
const int CATCH_ZONE_INIT     = 0;  // 初期状態
const int CATCH_ZONE_STAND_BY = 1;  // 使用可能
const int CATCH_ZONE_USED     = 2;  // 使用済み


/**
 * @brief 幅優先探索を利用して、あるマスから他のマスへの距離を計算する
 *
 * @tparam T 距離の型(マンハッタン距離)
 */
template<typename T>
struct DistCalclator {
  typedef pair<int, int> P;
  int h, w;
  vector<vector<bool>> blocked;
  vector<vector<T>> dist, move_rsv, stop_rsv;

  /**
   * @brief コンストラクタ
   *
   * @param _h 行数
   * @param _w 列数
   * @param _blocked 盤面(既に通行不能になっているマスの情報)
   * @param _move_rsv 盤面(人が移動する予定のマス情報)
   * @param _stop_rsv 盤面(人が通行不能にする予定のマス情報)
   */
  DistCalclator(int _h, int _w, const vector<vector<bool>>& _blocked, const vector<vector<T>>& _move_rsv, const vector<vector<T>>& _stop_rsv) {
    init(_h, _w, _blocked, _move_rsv, _stop_rsv);
  }

  /**
   * @brief 初期化
   *
   * @param _h 行数
   * @param _w 列数
   * @param _blocked 盤面(既に通行不能になっているマスの情報)
   * @param _move_rsv 盤面(人が移動する予定のマス情報)
   * @param _stop_rsv 盤面(人が通行不能にする予定のマス情報)
   */
  void init(int _h, int _w, const vector<vector<bool>>& _blocked, const vector<vector<T>>& _move_rsv, const vector<vector<T>>& _stop_rsv) {
    h = _h; w = _w; blocked = _blocked; move_rsv = _move_rsv; stop_rsv = _stop_rsv;
  }

  /**
   * @brief 距離テーブルの初期化
   */
  void reset() {
    dist.assign(h, vector<T>(w, INF));
  }

  /**
   * @brief 距離計算実行
   *
   * @param sy 始点の座標(行)
   * @param sx 始点の座標(列)
   * @param only_passage 通路のみ通行可能とするか
   */
  void run(int sy, int sx, bool only_passage=false) {
    int dy[] = {0, 0, 1, -1};
    int dx[] = {1, -1, 0, 0};
    reset();
    dist[sy][sx] = 0;
    queue<P> que;
    que.push(P(sy, sx));
    while (!que.empty()) {
      P p = que.front(); que.pop();
      int cy = p.first, cx = p.second;
      for (int i = 0; i < 4; i++) {
        int ny = cy + dy[i], nx = cx + dx[i];
        if (ny < 0 || h <= ny || nx < 0 || w <= nx) continue;
        if (blocked[ny][nx] || stop_rsv[ny][nx] > 0) continue;
        if (only_passage && base[ny][nx] != PASSAGE && base[ny][nx] != BLOCK_ACTION_POINT && base[ny][nx] != BLOCK_ACTION_POINT_SPE) continue;
        if (dist[ny][nx] <= dist[cy][cx] + 1) continue;

        dist[ny][nx] = dist[cy][cx] + 1;
        que.push(P(ny, nx));
      }
    }
  }

  /**
   * @brief 距離計算実行
   *
   * @param p 始点の座標
   * @param only_passage 通路のみ通行可能とするか
   */
  void run(const P& p, bool only_passage=false) {
    run(p.first, p.second, only_passage);
  }
};

/**
 * @brief マスa,b間のマンハッタン距離
 *
 * @param a
 * @param b
 * @return int マンハッタン距離
 */
int m_dist(const P a, const P b)
{
  return abs(a.first-b.first) + abs(a.second-b.second);
}

/**
 * @brief 最小費用流
 *
 * @tparam flow_t 流量の型
 * @tparam cost_t コストの型
 */
template<typename flow_t, typename cost_t>
struct PrimalDual {
  const cost_t INF;

  /**
   * @brief 辺
   */
  struct edge {
    int to;
    flow_t cap;
    cost_t cost;
    int rev;
    bool isrev;
  };
  vector<vector<edge>> graph;
  vector<cost_t> potential, min_cost;
  vector<int> prevv, preve;

  PrimalDual(int V) : graph(V), INF(numeric_limits<cost_t>::max()) {}

  /**
   * @brief 辺の追加
   *
   * @param from 始点
   * @param to 終点
   * @param cap 容量
   * @param cost コスト
   */
  void add_edge(int from, int to, flow_t cap, cost_t cost) {
    graph[from].emplace_back((edge){to, cap, cost, (int) graph[to].size(), false});
    graph[to].emplace_back((edge){from, 0, -cost, (int) graph[from].size()-1, true});
  }

  /**
   * @brief フロー f を流した時の最小費用流を計算する
   *
   * @param s 開始頂点
   * @param t 終了頂点
   * @param f フロー
   * @return cost_t 最小費用流
   */
  cost_t min_cost_flow(int s, int t, flow_t f) {
    int V = (int)graph.size();
    cost_t ret = 0;
    using P = pair<cost_t, int>;
    priority_queue<P, vector<P>, greater<P>> que;
    potential.assign(V, 0);
    prevv.assign(V, -1);
    preve.assign(V, -1);

    while (f > 0) {
      min_cost.assign(V, INF);
      que.emplace(0, s);
      min_cost[s] = 0;
      while (!que.empty()) {
        P p = que.top();
        que.pop();
        if (min_cost[p.second] < p.first) continue;
        for (int i = 0; i < graph[p.second].size(); i++) {
          edge &e = graph[p.second][i];
          cost_t next_cost = min_cost[p.second] + e.cost + potential[p.second] - potential[e.to];
          if (e.cap > 0 && min_cost[e.to] > next_cost) {
            min_cost[e.to] = next_cost;
            prevv[e.to] = p.second, preve[e.to] = i;
            que.emplace(min_cost[e.to], e.to);
          }
        }
      }

      if (min_cost[t] == INF) return -1;
      for (int v = 0; v < V; v++) potential[v] += min_cost[v];
      flow_t addflow = f;
      for (int v = t; v != s; v = prevv[v]) {
        addflow = min(addflow, graph[prevv[v]][preve[v]].cap);
      }
      f -= addflow;
      ret += addflow * potential[t];
      for (int v = t; v != s; v = prevv[v]) {
        edge &e = graph[prevv[v]][preve[v]];
        e.cap -= addflow;
        graph[v][e.rev].cap += addflow;
      }
    }

    return ret;
  }

  /**
   * @brief 計算結果出力
   */
  void output() {
    for (int i = 0; i < graph.size(); i++) {
      for (auto &e : graph[i]) {
        if (e.isrev) continue;
        auto &rev_e = graph[e.to][e.rev];
        cout << "# " << i << "->" << e.to << " (flow: " << rev_e.cap << "/" << rev_e.cap + e.cap << ")" << endl;
      }
    }
  }
};

/**
 * @brief ペットの移動
 *
 * @param i ペットのID
 * @param c 方向
 */
void pet_move(const int i, const char c)
{
  if (c < 'A' || 'Z' < c) return;

  P pre = pp[i];
  if (c == ACT_MOVE_UP)     pp[i].first--;
  if (c == ACT_MOVE_DOWN)   pp[i].first++;
  if (c == ACT_MOVE_LEFT)   pp[i].second--;
  if (c == ACT_MOVE_RIGHT)  pp[i].second++;

  pf[pre.first][pre.second] = pf[pre.first][pre.second] & ~(1<<i);
  pf[pp[i].first][pp[i].second] = pf[pp[i].first][pp[i].second] | (1<<i);
}

/**
 * @brief 人の移動
 *
 * @param i 人のID
 * @param c 方向
 */
void human_move(const int i, const char c)
{
  if (c < 'A' || 'Z' < c) return;

  P pre = hp[i];
  if (c == ACT_MOVE_UP)     hp[i].first--;
  if (c == ACT_MOVE_DOWN)   hp[i].first++;
  if (c == ACT_MOVE_LEFT)   hp[i].second--;
  if (c == ACT_MOVE_RIGHT)  hp[i].second++;

  hf[pre.first][pre.second] = hf[pre.first][pre.second] & ~(1<<i);
  hf[hp[i].first][hp[i].second] = hf[hp[i].first][hp[i].second] | (1<<i);
}

ll get_rand_range(ll min_val, ll max_val)
{
  if (min_val > max_val) swap(min_val, max_val);
  ll m = max_val - min_val + 1;
  return rand()%m + min_val;
  // static std::mt19937 mt((unsigned)time(NULL));
  // std::uniform_int_distribution<ll> get_rand_uni_int(min_val, max_val);
  // return get_rand_uni_int(mt);
}

// 隣のマスを通行不能にする (対象マスの隣に動物がいないこと前提)
/**
 * @brief 人が隣のマスを通行不能にする
 *
 * @param i 人のID
 * @param c 方向
 */
void stop_action(const int i, const char c)
{
  if (c < 'a' || 'z' < c) return;

  P p = hp[i];
  if (c == ACT_STOP_UP) {
    blocked[p.first-1][p.second] = true;
  }
  if (c == ACT_STOP_DOWN) {
    blocked[p.first+1][p.second] = true;
  }
  if (c == ACT_STOP_LEFT) {
    blocked[p.first][p.second-1] = true;
  }
  if (c == ACT_STOP_RIGHT) {
    blocked[p.first][p.second+1] = true;
  }
}

/**
 * @brief 動物の行動を実行する
 */
void pets_action()
{
  rep(i, N) {
    string s; cin >> s;
    for (auto &c:s) pet_move(i, c);
  }
}

/**
 * @brief 人の行動を実行する
 */
void humans_action()
{
  rep(i, M) {
    stop_action(i, act[i]);
    human_move(i, act[i]);
  }
}

/**
 * @brief データ準備
 */
void init()
{
  // 捕獲ゾーンのIDを振る
  vector<vector<bool>> visited(H, vector<bool>(W, false));
  queue<P> que;
  int id = 0;

  rep(i, H) rep(j, W) {

    if ((base[i][j] != CATCH_ZONE && base[i][j] != PRE_CATCH_ZONE) || visited[i][j]) continue;

    catch_zone_pos.push_back({i, j});
    catch_zone_status.push_back(0);
    que.push({i, j});
    while (!que.empty()) {
      P p = que.front(); que.pop();
      int cx = p.first, cy = p.second;
      catch_zone_id[cx][cy] = id;
      visited[cx][cy] = true;
      rep(k, 4) {
        int nx = cx + dx[k], ny = cy + dy[k];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        if (base[nx][ny] != CATCH_ZONE && base[nx][ny] != PRE_CATCH_ZONE && base[nx][ny] != BLOCK_POINT) continue;
        if (visited[nx][ny]) continue;
        que.push({nx, ny});
      }
    }
    id++;
  }

  // 最終作戦で待機するマスの候補
  for (int j=6; j<=23; j++) final_operation_pos.push_back({5, max(6, min(23, j))});
  for (int i=6; i<=23; i++) final_operation_pos.push_back({max(6, min(23, i)), 24});
  for (int j=23; j>=6; j--) final_operation_pos.push_back({24, max(6, min(22, j))});
  for (int i=23; i>=6; i--) final_operation_pos.push_back({max(6, min(23, i)), 5});

  // cout << "# catch zone id\n";
  // rep(i, H) {
  //   cout << "# " << catch_zone_id[i] << "\n";
  // }
}


/**
 * @brief マスpを通行不能にできるかどうかを調べる
 *
 * @param p 対象のマスの座標
 * @param move_rsv 人が移動する予定のマス情報
 * @param stop_rsv 人が通行不能にする予定のマス情報
 * @return true 通行不能にできる
 * @return false 通行不能にできない
 */
bool check_can_block(const P p, const vector<vector<int>>& move_rsv, const vector<vector<int>>& stop_rsv)
{
  int x = p.first, y = p.second;
  if (blocked[x][y]
      || pf[x][y] > 0 || hf[x][y] > 0                 // 動物か人がいる
      || move_rsv[x][y] > 0 || stop_rsv[x][y] > 0) {  // 他の人が移動もしくは通行不能にする予定である
    return false;
  }

  rep(i, 4) {
    int nx = x + dx[i], ny = y + dy[i];
    if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
    if (pf[nx][ny] > 0) return false;                 // 隣のマスに動物がいる
  }
  return true;
}

/**
 * @brief 捕獲ゾーンの状態を確認する
 *
 * @param i 捕獲ゾーンのID
 * @return vector<int> 捕獲できるペットのID 空の場合は捕獲ゾーン未完成もしくはペットが居ない
 */
vector<int> check_catch_zone_status(int i)
{
  assert(i < (int)catch_zone_pos.size());

  vector<vector<bool>> visited(H, vector<bool>(W, false));
  queue<P> que;
  que.push(catch_zone_pos[i]);
  vector<int> ret;
  bool used = true;

  while (!que.empty()) {
    P p = que.front(); que.pop();
    int cx = p.first, cy = p.second;
    visited[cx][cy] = true;
    if (hf[cx][cy] > 0) {
      return vector<int>(0);
    }

    if (pf[cx][cy] > 0) {
      int x = pf[cx][cy];
      int cnt = 0;
      while (x > 0) {
        if (x&1) ret.push_back(cnt);
        x /= 2;
        cnt++;
      }
    }

    rep(k, 4) {
      int nx = cx + dx[k], ny = cy + dy[k];
      if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
      if (visited[nx][ny]) continue;
      int t = base[nx][ny];
      if ((t == BLOCK || t == BLOCK_FOR_DOG) && !blocked[nx][ny]) {
        return vector<int>(0);
      }
      if (t == BLOCK_POINT && !blocked[nx][ny]) {
        used = false;
      }
      if (t != PRE_CATCH_ZONE && t != CATCH_ZONE) continue;
      que.push({nx, ny});
    }
  }
  return ret;
}

/**
 * @brief 移動予定の登録
 *
 * @param i 人のID
 * @param c 方向
 * @param move_rsv 既に決定済みの移動予定
 */
void reserve_move(const int& i, const char& c, vector<vector<int>>& move_rsv)
{
  act[i] = c;
  if (c < 'A' || 'Z' < c) return;

  if (c == ACT_MOVE_UP)     move_rsv[hp[i].first-1][hp[i].second] = 1;
  if (c == ACT_MOVE_DOWN)   move_rsv[hp[i].first+1][hp[i].second] = 1;
  if (c == ACT_MOVE_LEFT)   move_rsv[hp[i].first][hp[i].second-1] = 1;
  if (c == ACT_MOVE_RIGHT)  move_rsv[hp[i].first][hp[i].second+1] = 1;
}

void reserve_stop(const int& i, const char& c, vector<vector<int>>& stop_rsv)
{
  act[i] = c;
  if (c < 'a' || 'z' < c) return;

  if (c == ACT_STOP_UP)     stop_rsv[hp[i].first-1][hp[i].second] = 1;
  if (c == ACT_STOP_DOWN)   stop_rsv[hp[i].first+1][hp[i].second] = 1;
  if (c == ACT_STOP_LEFT)   stop_rsv[hp[i].first][hp[i].second-1] = 1;
  if (c == ACT_STOP_RIGHT)  stop_rsv[hp[i].first][hp[i].second+1] = 1;
}

/**
 * @brief フェーズ1 行動割当て
 */
void a_phase_1_asign()
{
  vector<int> used(M, false);
  int remains = M;

  // 地形作成要員を割当て
  {
    int offset = 10;
    int st = offset + (int)phase_1_startpos.size();
    int ed = st + 1;
    // PrimalDual<int, double> g(ed+1);
    PrimalDual<int, int> g(ed+1);

    rep(i, M) {
      g.add_edge(st, i, 1, 0);
      for (int j=0; j<(int)phase_1_startpos.size(); j++) {
        P &p = phase_1_startpos[j];
        // double xx = abs(hp[i].first - p.first);
        // double yy = abs(hp[i].second - p.second);
        // double cost = sqrt(xx*xx + yy*yy);
        int dist = abs(hp[i].first-p.first) + abs(hp[i].second-p.second);
        int cost = dist * dist;
        g.add_edge(i, j+offset, 1, cost);
      }
    }
    for (int i=0; i<(int)phase_1_startpos.size(); i++) {
      g.add_edge(i+offset, ed, 1, 0);
    }
    g.min_cost_flow(st, ed, min(M, (int)phase_1_startpos.size()));
    // g.output();

    rep(i, M) {
      for (auto &e : g.graph[i]) {
        if (e.isrev || e.to < offset || st <= e.to) continue;
        auto &rev_e = g.graph[e.to][e.rev];
        if (rev_e.cap < 1) continue;

        human[i].job = 1;
        human[i].meta = e.to - offset;
        human[i].subject_pos = phase_1_startpos[e.to - offset];
        used[i] = true;
        remains--;
      }
    }
  }

  // 余り要員を待機ポイントに割り当てる
  while (remains > 0) {
    int offset = 10;
    int st = offset + (int)phase_1_startpos.size();
    int ed = st + 1;
    // PrimalDual<int, double> g(ed+1);
    PrimalDual<int, int> g(ed+1);

    rep(i, M) {
      if (used[i]) continue;

      g.add_edge(st, i, 1, 0);
      for (int j=0; j<(int)phase_1_waitpos.size(); j++) {
        P &p = phase_1_waitpos[j];
        // double xx = abs(hp[i].first - p.first);
        // double yy = abs(hp[i].second - p.second);
        // double cost = sqrt(xx*xx + yy*yy);
        int dist = abs(hp[i].first-p.first) + abs(hp[i].second-p.second);
        int cost = dist * dist;
        g.add_edge(i, j+offset, 1, cost);
      }
    }
    for (int i=0; i<(int)phase_1_waitpos.size(); i++) {
      g.add_edge(i+offset, ed, 1, 0);
    }
    g.min_cost_flow(st, ed, min(remains, (int)phase_1_waitpos.size()));
    // g.output();

    rep(i, M) {
      for (auto &e : g.graph[i]) {
        if (e.isrev || e.to < offset || st <= e.to) continue;
        auto &rev_e = g.graph[e.to][e.rev];
        if (rev_e.cap < 1) continue;

        human[i].job = 3;
        human[i].meta = e.to - offset;
        human[i].subject_pos = phase_1_waitpos[e.to - offset];
        used[i] = true;
        remains--;
      }
    }
  }
}

double calc_score(const vector<vector<int>> &stop_rsv)
{
  double ret = 0;

  vector<vector<bool>> visited(H, vector<bool>(W, false));
  rep(i, H) rep(j, W) {
    if (visited[i][j]) continue;

    int pet_cnt = 0;
    double human_cnt = 0;
    double panel_cnt = 0;
    queue<P> que;
    que.push({i, j});
    while (!que.empty()) {
      P p = que.front(); que.pop();
      int cx = p.first, cy = p.second;
      if (visited[cx][cy]) continue;
      visited[cx][cy] = true;
      pet_cnt += pcnt(pf[cx][cy]);
      human_cnt += pcnt(hf[cx][cy]);
      panel_cnt += 1;
      rep(k, 4) {
        int nx = cx + dx[k], ny = cy + dy[k];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        if (visited[nx][ny] || blocked[nx][ny] || stop_rsv[nx][ny]>0) continue;
        que.push({nx, ny});
      }
    }

    double tmp = human_cnt * panel_cnt * 100000000;
    while (pet_cnt--) tmp /= 2;
    ret += tmp;
  }

  return ret;
}

P dest_pos(const P p, char c)
{
  P ret = p;
  if (c == ACT_STOP_UP) ret.first--;
  if (c == ACT_STOP_DOWN) ret.first++;
  if (c == ACT_STOP_LEFT) ret.second--;
  if (c == ACT_STOP_RIGHT) ret.second++;
  if (c == ACT_MOVE_UP) ret.first--;
  if (c == ACT_MOVE_DOWN) ret.first++;
  if (c == ACT_MOVE_LEFT) ret.second--;
  if (c == ACT_MOVE_RIGHT) ret.second++;
  return ret;
}

/**
 * @brief 移動方向を決める
 *
 * @param i 人のID
 * @param d 距離計算
 * @param type 0:縦方向優先, 1:横方向優先
 * @param only_passage 通路のみ通行可能とするか
 * @return char 移動方向(U,D,L,R) 移動しないまたはできない場合は'.'
 */
char decide_move_dir(const int i, DistCalclator<int> &d, const int type = 0, const bool only_passage = false)
{
  // 目的地からの距離を計算する
  d.run(human[i].subject_pos, only_passage);

  char ret = ACT_NONE;

  int cx = hp[i].first, cy = hp[i].second;
  rep(j, 4) {
    int k = j;
    if (type == 1) k = (k+2)%4;
    int nx = cx + dx[k], ny = cy + dy[k];
    if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
    if (d.dist[cx][cy] <= d.dist[nx][ny]) continue;

    ret = ACT_MOVE[k];
    break;
  }

  return ret;
}

/**
 * @brief 通行不能にする方向を決める
 *
 * @param i 人のID
 * @param move_rsv 人の移動先予定
 * @param stop_rsv 人が通行不能にする予定
 * @param bits 通行不能にしたくない方向をビット集合で指定
 *              1ビット目上、2ビット目下、3ビット目左、4ビット目右
 * @return pair<char, bool> char:通行不能にする方向(u,d,l,r)
 *                          bool:通行不能にしたいマスがある場合true
 */
pair<char, bool> decide_block_dir(const int i, vector<vector<int>> &move_rsv, vector<vector<int>> &stop_rsv, const int bits = 0, const bool for_dog = false)
{
  char ret = ACT_NONE;
  bool exists = false;

  int cx = hp[i].first, cy = hp[i].second;
  rep(j, 4) {
    if (bits>>j&1) continue;
    int nx = cx + dx[j], ny = cy + dy[j];
    if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
    if (base[nx][ny] != (for_dog?BLOCK_FOR_DOG:BLOCK)) continue;
    if (blocked[nx][ny]) continue;
    exists = true;
    if (!check_can_block({nx, ny}, move_rsv, stop_rsv)) continue;
    ret = ACT_STOP[j];
    break;
  }

  return {ret, exists};
}

/**
 * @brief 最も近い通路の座標
 *
 * @param p
 * @param stop_rsv 通行止め予定
 * @return P
 */
P nearest_passage(const P p, const vector<vector<int>>& stop_rsv)
{
  if (base[p.first][p.second] == PASSAGE || base[p.first][p.second] == BLOCK_ACTION_POINT || base[p.first][p.second] == BLOCK_ACTION_POINT_SPE) {
    return p;
  }

  vector<vector<int>> dist(H, vector<int>(W, INF));
  dist[p.first][p.second] = 0;
  queue<P> que;
  que.push(p);

  while (!que.empty()) {
    P p = que.front(); que.pop();
    int cx = p.first, cy = p.second;
    for (int k=0; k<4; k++) {
      int nx = cx + dx[k], ny = cy + dy[k];
      if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
      if (base[nx][ny] == BLOCK || blocked[nx][ny] || stop_rsv[nx][ny] > 0) continue;
      if (dist[nx][ny] <= dist[cx][cy] + 1) continue;
      dist[nx][ny] = dist[cx][cy] + 1;
      int b = base[nx][ny];
      if (b == PASSAGE || b == BLOCK_ACTION_POINT || b == BLOCK_ACTION_POINT_SPE) {
        return {nx, ny};
      }
      que.push({nx, ny});
    }
  }
  return {10, 10};
}

/**
 * @brief フェーズ1 移動～構築
 */
void a_phase_1_build()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    if (human[i].job == 1) {
      // 目的地に着いたらjob2に移行
      if (hp[i] == human[i].subject_pos) {
        human[i].job = 2;
        human[i].subject_pos = phase_1_waitpos[human[i].meta];
        i--; continue;
      }

      char c = decide_move_dir(i, d, human[i].meta&1);
      reserve_move(i, c, move_rsv);
      continue;
    }

    if (human[i].job == 2) {
      // 目的地に着いたらjob3に移行
      if (hp[i] == human[i].subject_pos) {
        human[i].job = 3;
        i--; continue;
      }

      auto p = decide_block_dir(i, move_rsv, stop_rsv, 0, true);
      char c = p.first;
      reserve_stop(i, c, stop_rsv);
      if (c == ACT_NONE && !p.second) {
        c = decide_move_dir(i, d, (human[i].meta/2)^1);
        reserve_move(i, c, move_rsv);
      }

      continue;
    }

    if (human[i].job == 3) {
      char c = decide_move_dir(i, d);
      reserve_move(i, c, move_rsv);
      bool next_phase = false;
      if (c == ACT_NONE) {  // 到着済み
        auto p = decide_block_dir(i, move_rsv, stop_rsv, 0, true);
        reserve_stop(i, p.first, stop_rsv);
        next_phase = !p.second;
      }
      if (next_phase) {
        human[i].job = 4;
      }
    }
  }
}

bool check_all_dog_in_catchzone()
{
  bool ret = true;
  rep(i, N) {
    if (pt[i] != DOG) continue;
    if (base[pp[i].first][pp[i].second] != DOG_CATCH_ZONE) ret = false;
  }

  return ret;
}

bool is_phase_2_2 = false;

void a_phase_2_1()
{
  if (!check_all_dog_in_catchzone()) return;

  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  vector<bool> can_block(4, false);
  bool isok = true;
  string pre_act = "";
  rep(i, M) {
    pre_act.push_back(ACT_NONE);
    int k = human[i].meta;
    if (can_block[k]) continue;
    int nx = hp[i].first + dx[k], ny = hp[i].second + dy[k];
    // cout << "# k=" << k << " , nx=" << nx << " , ny=" << ny << "\n";
    if (check_can_block({nx, ny}, move_rsv, stop_rsv)) {
      can_block[k] = true;
      pre_act[i] = ACT_STOP[k];
    } else {
      isok = false;
    }
  }

  if (isok) {
    act = pre_act;
    phase = PHASE_3;
    int dog_catch_turn = 300-turn;
    debug(dog_catch_turn);
    return;
  }

  rep(i, M) {
    char c = ACT_MOVE[human[i].meta^1];
    reserve_move(i, c, move_rsv);
  }
  is_phase_2_2 = true;
}

void a_phase_2_2()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  if (!check_all_dog_in_catchzone()) {
    rep(i, M) {
      char c = ACT_MOVE[human[i].meta];
      reserve_move(i, c, move_rsv);
    }
    is_phase_2_2 = false;
    return;
  }

  vector<bool> can_block(4, false);
  bool isok = true;
  string pre_act = "";
  rep(i, M) {
    // cout << "# i=" << i+1 << ", job=" << human[i].job << "\n";
    pre_act.push_back(ACT_NONE);
    int k = human[i].meta;
    if (can_block[k]) continue;
    int nx = hp[i].first + dx[k], ny = hp[i].second + dy[k];
    // cout << "# i=" << i+1 << ", k=" << k << " , nx=" << nx << " , ny=" << ny << "\n";
    if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
    if (check_can_block({nx, ny}, move_rsv, stop_rsv)) {
      can_block[k] = true;
      pre_act[i] = ACT_STOP[k];
    } else {
      isok = false;
    }
  }

  if (isok) {
    act = pre_act;
    phase = PHASE_3;
    int dog_catch_turn = 300-turn;
    debug(dog_catch_turn);
    return;
  }

  rep(i, M) {
    char c = ACT_MOVE[human[i].meta];
    reserve_move(i, c, move_rsv);
  }
  is_phase_2_2 = false;
}

/**
 * @brief 犬閉じ込め作戦
 */
void a_phase_2()
{
  if (is_phase_2_2) {
    a_phase_2_2();
    return;
  }

  a_phase_2_1();
}

void a_phase_3_asign()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

  int center = 2;
  if (M >= 7) center++;
  int other = M - center;
  // cout << "# a_phase_3_asign start\n";

  // 外周建築要員割当て
  {
    // cout << "# periphery asign start\n";
    int offset = 10;
    int st = offset + other;
    int ed = st + 1;

    int mi = INF;
    int siz = build_sequence_periphery.size();
    int drt = siz / other;

    for (int k=0; k<drt; k++) {
      PrimalDual<int, int> g(ed+1);
      rep(i, M) {
        d.run(hp[i]);
        g.add_edge(st, i, 1, 0);
        for (int j=0; j<other; j++) {
          P &p = build_sequence_periphery[k+j*drt].first;
          int dist = d.dist[p.first][p.second];
          int cost = dist * dist;
          g.add_edge(i, j+offset, 1, cost);
        }
      }
      for (int i=0; i<other; i++) {
        g.add_edge(i+offset, ed, 1, 0);
      }
      int mcf = g.min_cost_flow(st, ed, other);
      if (mi > mcf) {
        mi = mcf;
        rep(i, M) {
          human[i].job = 0;
          human[i].meta = 0;
          human[i].subject_pos = {5, 5};

          for (auto &e : g.graph[i]) {
            if (e.isrev || e.to < offset || st <= e.to) continue;
            auto &rev_e = g.graph[e.to][e.rev];
            if (rev_e.cap < 1) continue;
            int tmp = k + (e.to - offset)*drt;

            human[i].job = 1;
            human[i].subject_pos = build_sequence_periphery[k + (e.to - offset)*drt].first;
          }
        }
      }
    }
  }

  vector<int> r;
  rep(i, M) {
    if (human[i].job == 0) r.push_back(i);
  }

  // 中央部建築要員割当て
  {
    // cout << "# center asign start\n";
    int offset = 10;
    int st = offset + center;
    int ed = st + 1;

    int mi = INF;
    int siz = build_sequence_center.size();
    int drt = siz / center;

    for (int k=0; k<drt; k++) {
      PrimalDual<int, int> g(ed+1);
      rep(i, M) {
        if (human[i].job == 1) continue;

        d.run(hp[i]);
        g.add_edge(st, i, 1, 0);
        for (int j=0; j<center; j++) {
          P &p = build_sequence_center[k+j*drt].first;
          int dist = d.dist[p.first][p.second];
          int cost = dist * dist;
          g.add_edge(i, j+offset, 1, cost);
        }
      }
      for (int i=0; i<center; i++) {
        g.add_edge(i+offset, ed, 1, 0);
      }
      int mcf = g.min_cost_flow(st, ed, center);
      if (mi > mcf) {
        mi = mcf;
        rep(i, M) {
          if (human[i].job == 1) continue;

          for (auto &e : g.graph[i]) {
            if (e.isrev || e.to < offset || st <= e.to) continue;
            auto &rev_e = g.graph[e.to][e.rev];
            if (rev_e.cap < 1) continue;

            human[i].job = 3;
            human[i].subject_pos = build_sequence_center[k + (e.to - offset)*drt].first;
          }
        }
      }
    }
  }

  // rep(i, M) {
  //   debug(i, human[i].job, human[i].subject_pos);
  //   cout << "# " << i+1 << ": job=" << human[i].job << ", subject_pos=" << human[i].subject_pos.first << "," << human[i].subject_pos.second << "\n";
  // }
}

void a_phase_3_build()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    if (human[i].job == 1) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }

      for (int j=0; j<(int)build_sequence_periphery.size(); j++) {
        if (build_sequence_periphery[j].first == hp[i]) {
          human[i].job = 2;
          human[i].meta = j;
          break;
        }
      }
    }

    if (human[i].job == 3) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }

      for (int j=0; j<(int)build_sequence_center.size(); j++) {
        if (build_sequence_center[j].first == hp[i]) {
          human[i].job = 4;
          human[i].meta = j;
          break;
        }
      }
    }


    if (human[i].job == 2) {
      int k = human[i].meta;
      while (build_sequence_periphery[k].second < 'a' || 'z' < build_sequence_periphery[k].second) {
        k = (k+1) % (int)build_sequence_periphery.size();
      }
      P p = dest_pos(build_sequence_periphery[k].first, build_sequence_periphery[k].second);
      // cout << "# " << i+1 << " : phase_1 job 2 " << human[i].meta << "\n";
      // cout << "# pos=" << hp[i].first << "," << hp[i].second << ", next action is " << build_sequence_periphery[human[i].meta].second << "\n";
      // cout << "# p=" << p.first << "," << p.second << "\n";
      if (!blocked[p.first][p.second]) {
        char c = build_sequence_periphery[human[i].meta].second;
        P dest = dest_pos(hp[i], c);
        if ('a' <= c && c <= 'z') {
          if (check_can_block(dest, move_rsv, stop_rsv)) {
            act[i] = c;
            reserve_stop(i, c, stop_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_periphery.size();
          }
        }
        if ('A' <= c && c <= 'Z') {
          if (!blocked[dest.first][dest.second] && stop_rsv[dest.first][dest.second] <= 0) {
            act[i] = c;
            reserve_move(i, c, move_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_periphery.size();
          }
        }

        continue;
      }

      human[i].job = 41;
      human[i].subject_pos = nearest_passage(hp[i], stop_rsv);
      human[i].meta = -1;
    }

    if (human[i].job == 4) {
      int k = human[i].meta;
      while (build_sequence_center[k].second < 'a' || 'z' < build_sequence_center[k].second) {
        k = (k+1) % (int)build_sequence_center.size();
      }
      P p = dest_pos(build_sequence_center[k].first, build_sequence_center[k].second);
      // cout << "# " << i+1 << " : phase_1 job 4 " << human[i].meta << "\n";
      // cout << "# pos=" << hp[i].first << "," << hp[i].second << ", next action is " << build_sequence_center[human[i].meta].second << "\n";
      // cout << "# p=" << p.first << "," << p.second << "\n";
      if (!blocked[p.first][p.second]) {
        char c = build_sequence_center[human[i].meta].second;
        P dest = dest_pos(hp[i], c);
        if ('a' <= c && c <= 'z') {
          if (check_can_block(dest, move_rsv, stop_rsv)) {
            act[i] = c;
            reserve_stop(i, c, stop_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_center.size();
          }
        }
        if ('A' <= c && c <= 'Z') {
          if (!blocked[dest.first][dest.second] && stop_rsv[dest.first][dest.second] <= 0) {
            act[i] = c;
            reserve_move(i, c, move_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_center.size();
          }
        }

        continue;
      }

      human[i].job = 41;
      human[i].subject_pos = nearest_passage(hp[i], stop_rsv);
      human[i].meta = -1;
    }

    // if (human[i].job == 3) {
    //   if (hp[i] != human[i].subject_pos) {
    //     char c = decide_move_dir(i, d);
    //     reserve_move(i, c, move_rsv);
    //   }
    //   continue;
    // }
  }

}

void plan_a()
{
  rep(i, M) act[i] = ACT_NONE;

  if (phase == PHASE_0) {
    a_phase_1_asign();
    phase = PHASE_1;
  }

  if (phase == PHASE_1) {
    bool next_phase = true;
    rep(i, M) {
      if (human[i].job < 4) next_phase = false;
    }
    if (next_phase) {
      phase = PHASE_2;
    } else {
      a_phase_1_build();
      return;
    }
  }

  if (phase == PHASE_2) {
    a_phase_2();
    if (phase == PHASE_3) a_phase_3_asign();
    return;
  }

  if (phase == PHASE_3) {
    a_phase_3_build();
  }
}

void b_phase_1_asign()
{
  int center = 2;
  int other = M - center;

  // 外周建築要員の割当て
  {
    int offset = 10;
    int st = offset + other;
    int ed = st + 1;

    int mi = INF;
    int siz = build_sequence_periphery.size();
    int drt = siz / other;

    for (int k=0; k<drt; k++) {
      PrimalDual<int, int> g(ed+1);
      rep(i, M) {
        g.add_edge(st, i, 1, 0);
        for (int j=0; j<other; j++) {
          P &p = build_sequence_periphery[k+j*drt].first;
          int dist = abs(hp[i].first-p.first) + abs(hp[i].second-p.second);
          int cost = dist * dist;
          g.add_edge(i, j+offset, 1, cost);
        }
      }
      for (int i=0; i<other; i++) {
        g.add_edge(i+offset, ed, 1, 0);
      }
      int mcf = g.min_cost_flow(st, ed, other);
      if (mi > mcf) {
        mi = mcf;
        rep(i, M) {
          human[i].job = 0;
          human[i].meta = 0;
          human[i].subject_pos = {14, 14};

          for (auto &e : g.graph[i]) {
            if (e.isrev || e.to < offset || st <= e.to) continue;
            auto &rev_e = g.graph[e.to][e.rev];
            if (rev_e.cap < 1) continue;

            human[i].job = 1;
            human[i].subject_pos = build_sequence_periphery[k + (e.to - offset)*drt].first;
          }
        }
      }
    }
  }

  vector<int> r;
  rep(i, M) {
    if (human[i].job == 0) r.push_back(i);
  }

  // 中央部建築要員割当て
  {
    // cout << "# center asign start\n";
    int offset = 10;
    int st = offset + center;
    int ed = st + 1;

    int mi = INF;
    int siz = build_sequence_center_nodog.size();
    int drt = siz / center;

    for (int k=0; k<drt; k++) {
      PrimalDual<int, int> g(ed+1);
      rep(i, M) {
        if (human[i].job == 1) continue;

        g.add_edge(st, i, 1, 0);
        for (int j=0; j<center; j++) {
          P &p = build_sequence_center_nodog[k+j*drt].first;
          int dist = abs(hp[i].first-p.first) + abs(hp[i].second-p.second);
          int cost = dist * dist;
          g.add_edge(i, j+offset, 1, cost);
        }
      }
      for (int i=0; i<center; i++) {
        g.add_edge(i+offset, ed, 1, 0);
      }
      int mcf = g.min_cost_flow(st, ed, center);
      if (mi > mcf) {
        mi = mcf;
        rep(i, M) {
          if (human[i].job == 1) continue;

          for (auto &e : g.graph[i]) {
            if (e.isrev || e.to < offset || st <= e.to) continue;
            auto &rev_e = g.graph[e.to][e.rev];
            if (rev_e.cap < 1) continue;

            human[i].job = 21;
            human[i].subject_pos = build_sequence_center_nodog[k + (e.to - offset)*drt].first;
          }
        }
      }
    }
  }

  // デバッグ用出力 割当て結果
  // rep(i, M) {
  //   cout << "# " << i+1 << ": job=" << human[i].job << ", subject_pos=" << human[i].subject_pos << ", meta=" << human[i].meta << "\n";
  // }
}

void b_phase_1_build()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    if (human[i].job == 1) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }

      for (int j=0; j<(int)build_sequence_periphery.size(); j++) {
        if (build_sequence_periphery[j].first == hp[i]) {
          human[i].job = 2;
          human[i].meta = j;
          break;
        }
      }
    }

    if (human[i].job == 21) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }

      for (int j=0; j<(int)build_sequence_center_nodog.size(); j++) {
        if (build_sequence_center_nodog[j].first == hp[i]) {
          human[i].job = 22;
          human[i].meta = j;
          break;
        }
      }
    }

    // if (human[i].job == 11 || human[i].job == 21 || human[i].job == 31) {
    //   if (hp[i] != human[i].subject_pos) {
    //     char c = decide_move_dir(i, d);
    //     reserve_move(i, c, move_rsv);
    //     continue;
    //   }

    //   human[i].job++;
    //   human[i].meta = 0;
    // }

    if (human[i].job == 2) {
      int k = human[i].meta;
      while (build_sequence_periphery[k].second < 'a' || 'z' < build_sequence_periphery[k].second) {
        k = (k+1) % (int)build_sequence_periphery.size();
      }
      P p = dest_pos(build_sequence_periphery[k].first, build_sequence_periphery[k].second);
      // cout << "# " << i+1 << " : phase_1 job 2 " << human[i].meta << "\n";
      // cout << "# pos=" << hp[i].first << "," << hp[i].second << ", next action is " << build_sequence_periphery[human[i].meta].second << "\n";
      // cout << "# p=" << p.first << "," << p.second << "\n";
      if (!blocked[p.first][p.second]) {
        char c = build_sequence_periphery[human[i].meta].second;
        P dest = dest_pos(hp[i], c);
        if ('a' <= c && c <= 'z') {
          if (check_can_block(dest, move_rsv, stop_rsv)) {
            act[i] = c;
            reserve_stop(i, c, stop_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_periphery.size();
          }
        }
        if ('A' <= c && c <= 'Z') {
          if (!blocked[dest.first][dest.second] && stop_rsv[dest.first][dest.second] <= 0) {
            act[i] = c;
            reserve_move(i, c, move_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_periphery.size();
          }
        }

        continue;
      }

      human[i].job = 41;
      human[i].subject_pos = nearest_passage(hp[i], stop_rsv);
      human[i].meta = -1;
    }

    if (human[i].job == 3) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
      }
      continue;
    }

    // 中央外部建築
    if (human[i].job == 22) {
      int k = human[i].meta;
      while (build_sequence_center_nodog[k].second < 'a' || 'z' < build_sequence_center_nodog[k].second) {
        k = (k+1) % (int)build_sequence_center_nodog.size();
      }
      P p = dest_pos(build_sequence_center_nodog[k].first, build_sequence_center_nodog[k].second);

      if (!blocked[p.first][p.second]) {
        char c = build_sequence_center_nodog[human[i].meta].second;
        P dest = dest_pos(hp[i], c);
        // cout << "# " << i+1 << " : phase_1 job 22 " << human[i].meta << "\n";
        // cout << "# pos=" << hp[i].first << "," << hp[i].second << ", next action is " << build_sequence_periphery[human[i].meta].second << "\n";
        // cout << "# p=" << p.first << "," << p.second << "\n";
        if ('a' <= c && c <= 'z') {
          if (check_can_block(dest, move_rsv, stop_rsv)) {
            act[i] = c;
            reserve_stop(i, c, stop_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_center_nodog.size();
          }
        }
        if ('A' <= c && c <= 'Z') {
          if (!blocked[dest.first][dest.second] && stop_rsv[dest.first][dest.second] <= 0) {
            act[i] = c;
            reserve_move(i, c, move_rsv);
            human[i].meta = (human[i].meta+1) % (int)build_sequence_center_nodog.size();
          }
        }
        continue;
      }

      human[i].job = 41;
      human[i].subject_pos = nearest_passage(hp[i], stop_rsv);
      human[i].meta = -1;
    }

  }
}

/**
 * @brief ペットから見た最も近い通路マスを更新する
 *
 * @return P
 */
void update_nearest_passage_from_pets()
{
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, N) {
    pnp[i] = nearest_passage(pp[i], stop_rsv);
    if (pnp[i] == P{10, 10}) ps[i] = true;
    else ps[i] = false;
  }
}

/**
 * @brief 捕獲目標を更新する
 *
 * @param i 人のID
 */
void update_target_bk(int i)
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  if (human[i].job != 42) return;
  human[i].meta = -1;
  DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);
  d.run(hp[i], true);

  int mi = INF;
  vector<int> rem;
  rep(j, N) {
    if (ps[j] > 0 || pnp[j] == P{10, 10}) continue;
    bool isok = true;
    rep(k, M) {
      if (human[k].job == 42 && human[k].meta == j) isok = false;
    }
    if (!isok) continue;
    rem.push_back(j);

    if (mi > d.dist[pnp[j].first][pnp[j].second]) {
      mi = d.dist[pnp[j].first][pnp[j].second];
      human[i].meta = j;
      human[i].subject_pos = pnp[j];
    }
  }

  if (human[i].meta >= 0 || rem.empty()) return;

  int r = get_rand_range(0, (int)rem.size()-1);
  int j = rem[r];
  human[i].meta = j;
  human[i].subject_pos = pnp[j];
}

void update_target()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

  int pet_cnt = 0;
  int human_cnt = 0;

  int offset = 10;
  int st = offset + 20;
  int ed = st + 1;

  PrimalDual<int, int> g(ed+1);
  rep(i, M) {
    if (human[i].job != 42 && human[i].job != 52) continue;
    human[i].meta = -1;
    human_cnt++;
    d.run(hp[i], true);
    g.add_edge(st, i, 1, 0);

    rep(j, N) {
      if (ps[j]) continue;

      P &p = pnp[j];
      int dist = d.dist[p.first][p.second];
      if (dist >= INF) continue;
      int cost = dist * dist;
      if (pt[j] == CAT) cost /= 2;
      g.add_edge(i, j+offset, 1, cost);
    }
  }
  rep(j, N) {
    if (ps[j]) continue;
    pet_cnt++;
    g.add_edge(j+offset, ed, 1, 0);
  }
  int mcf = g.min_cost_flow(st, ed, min(human_cnt, pet_cnt));

  rep(i, M) {
    for (auto &e : g.graph[i]) {
      if (e.isrev || e.to < offset || st <= e.to) continue;
      auto &rev_e = g.graph[e.to][e.rev];
      if (rev_e.cap < 1) continue;

      int t = e.to - offset;
      human[i].meta = t;
      human[i].subject_pos = pnp[t];
    }
  }

  // rep(i, M) {
  //   cout << "# " << i+1 << ": job=" << human[i].job << ", meta=" << human[i].meta << ", subject_pos=" << human[i].subject_pos << "\n";
  // }
}

/**
 * @brief ペット捕獲作戦
 *
 */
void catching_pets_operation()
{
  update_nearest_passage_from_pets();
  // cout << "# pnps = ";
  // rep(i, N) cout << pnp[i] << ", ";
  // cout << "\n";

  // 建築が終わっているか確認する
  bool build_completed = true;
  rep(i, M) {
    if (human[i].job < 40) build_completed = false;
  }
  // 残りターン数が少なくなったら最終作戦を実行する
  bool is_pet_remains = false;
  rep(i, N) if (!ps[i]) is_pet_remains = true;
  if (is_pet_remains && build_completed && turn < FINAL_OPERATION_TURN) {
    phase = PHASE_8;
    return;
  }

  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    // 通路に向かって移動
    if (human[i].job == 41) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }

      human[i].job = 42;
    }
  }

  update_target();

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    // 標的を捕獲するための行動を行う
    if (human[i].job == 42) {
      // update_target_bk(i);
      // if (human[i].meta < 0) continue;

      // 捕獲ポイントに立っていて、かつたまたまいずれかのペットが捕獲可能な状態である場合は捕獲する
      int cx = hp[i].first, cy = hp[i].second;
      if (base[cx][cy] == BLOCK_ACTION_POINT) {
        // cout << "# now " << i+1 << " is on block action point.\n";
        int czid = -1;
        P dest = {10, 10};
        rep(k, 4) {
          if (act[i] != ACT_NONE) break;

          int nx = cx + dx[k], ny = cy + dy[k];
          if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
          if (base[nx][ny] != BLOCK_POINT) continue;
          // cout << "# k=" << k << " {nx,ny}=" << nx<<","<<ny << " base=" << base[nx][ny] << " czid=" << catch_zone_id[nx][ny] << "\n";
          czid = catch_zone_id[nx][ny];
          if (czid >= 0) {
            char c = ACT_STOP[k];
            dest = dest_pos(hp[i], c);
            vector<int> pets = check_catch_zone_status(czid);
            bool isok = check_can_block(dest, move_rsv, stop_rsv);
            // cout << "# c = " << c <<  " pets = " << pets << " isok = " << isok << "\n";
            if (!pets.empty() && isok) {
              act[i] = c;
              reserve_stop(i, c, stop_rsv);
              // for (auto val:pets) ps[val] = true;
            }
          }
        }
      }

      if (human[i].meta < 0 || act[i] != ACT_NONE || hp[i] == human[i].subject_pos) continue;

      char c = decide_move_dir(i, d, 0, true);
      // cout << "# i=" << i+1 << ", c=" << c << ", pnp=" << pnp[human[i].meta] << ", subject_pos=" << human[i].subject_pos << "\n";
      act[i] = c;
      reserve_move(i, c, move_rsv);
    }
  }
}

void final_operation_asign()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));
  DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

  int offset = 10;
  int st = offset + M;
  int ed = st + 1;

  int mi = INF;
  int siz = final_operation_pos.size();
  int drt = siz / M;

  for (int k=0; k<drt; k++) {
    PrimalDual<int, int> g(ed+1);
    rep(i, M) {
      d.run(hp[i]);
      g.add_edge(st, i, 1, 0);
      for (int j=0; j<M; j++) {
        P &p = final_operation_pos[k+j*drt];
        int dist = d.dist[p.first][p.second];
        int cost = dist * dist;
        g.add_edge(i, j+offset, 1, cost);
      }
    }
    for (int i=0; i<M; i++) {
      g.add_edge(i+offset, ed, 1, 0);
    }
    int mcf = g.min_cost_flow(st, ed, M);
    if (mi > mcf) {
      mi = mcf;
      rep(i, M) {
        human[i].job = 0;
        human[i].meta = 0;
        human[i].subject_pos = {5, 5};

        for (auto &e : g.graph[i]) {
          if (e.isrev || e.to < offset || st <= e.to) continue;
          auto &rev_e = g.graph[e.to][e.rev];
          if (rev_e.cap < 1) continue;
          int tmp = k + (e.to - offset)*drt;

          human[i].job = 51;
          human[i].subject_pos = final_operation_pos[k + (e.to - offset)*drt];
        }
      }
    }
  }

  phase = PHASE_9;
  // rep(i, M) {
  //   debug(i, human[i].job, human[i].subject_pos);
  //   cout << "# " << i+1 << ": job=" << human[i].job << ", subject_pos=" << human[i].subject_pos << "\n";
  // }
}

void final_operation_prepare()
{
  // cout << "# final_operation_prepare start\n";
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  rep(i, M) {
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    if (human[i].job == 51) {
      if (hp[i] != human[i].subject_pos) {
        char c = decide_move_dir(i, d);
        reserve_move(i, c, move_rsv);
        continue;
      }
      human[i].job = 52;
    }
  }

  // 全員指定位置に到着するまで待つ
  bool isok = true;
  rep(i, M) {
    if (human[i].job < 52) isok = false;
    else human[i].job = 52;
  }
  if (!isok) return;


  double ma = calc_score(stop_rsv);
  P a, b;
  isok = false;
  rep(ia, M) rep(ib, ia) {
    if (ia == ib) continue;
    rep(ka, 4) rep(kb, 4) {
      rep(kka, 4) if (ka != kka) {
        int nx = hp[ia].first + dx[kka], ny = hp[ia].second + dy[kka];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        stop_rsv[nx][ny]++;
      }
      rep(kkb, 4) if (kb != kkb) {
        int nx = hp[ib].first + dx[kkb], ny = hp[ib].second + dy[kkb];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        stop_rsv[nx][ny]++;
      }
      double tmp = calc_score(stop_rsv);
      rep(kka, 4) if (ka != kka) {
        int nx = hp[ia].first + dx[kka], ny = hp[ia].second + dy[kka];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        stop_rsv[nx][ny]--;
      }
      rep(kkb, 4) if (kb != kkb) {
        int nx = hp[ib].first + dx[kkb], ny = hp[ib].second + dy[kkb];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        stop_rsv[nx][ny]--;
      }
      if (ma < tmp) {
        isok = true;
        ma = tmp;
        a = {ia, ka}, b = {ib, kb};
      }
    }
  }

  if (isok) {
    human[a.first].job = 53;
    human[a.first].meta = a.second;
    human[b.first].job = 53;
    human[b.first].meta = b.second;
  }


  update_target();
  // phase = PHASE_10;
  // debug(ma, a, b);
}
void final_operation_action()
{
  vector<vector<int>> move_rsv(H, vector<int>(W, 0));
  vector<vector<int>> stop_rsv(H, vector<int>(W, 0));

  // rep(i, M) {
  //   if (human[i].job == 53) {
  //     rep(k, 4) {
  //       int t = k;
  //       if (human[i].meta >= 2) t = (t+2)%4;
  //       if (human[i].meta == t) continue;
  //       char c = ACT_STOP[t];
  //       P p = dest_pos(hp[i], c);
  //       if (check_can_block(p, move_rsv, stop_rsv)) {
  //         act[i] = c;
  //         reserve_stop(i, c, stop_rsv);
  //         break;
  //       }
  //     }
  //   }
  // }

  vector<int> cnt(M, 0);
  vector<int> rem(M, 0);
  bool is_last_one = true;

  rep(i, M) {
    if (human[i].job != 52) continue;
    DistCalclator<int> d(H, W, blocked, move_rsv, stop_rsv);

    // 捕獲ポイントに立っていて、かつたまたまいずれかのペットが捕獲可能な状態である場合は捕獲する
    int cx = hp[i].first, cy = hp[i].second;
    if (base[cx][cy] == BLOCK_ACTION_POINT) {
      // cout << "# now " << i+1 << " is on block action point.\n";
      int czid = -1;
      P dest = {10, 10};
      rep(k, 4) {
        if (act[i] != ACT_NONE) break;

        int nx = cx + dx[k], ny = cy + dy[k];
        if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
        if (base[nx][ny] != BLOCK_POINT) continue;
        // cout << "# k=" << k << " {nx,ny}=" << nx<<","<<ny << " base=" << base[nx][ny] << " czid=" << catch_zone_id[nx][ny] << "\n";
        czid = catch_zone_id[nx][ny];
        if (czid >= 0) {
          char c = ACT_STOP[k];
          dest = dest_pos(hp[i], c);
          vector<int> pets = check_catch_zone_status(czid);
          bool isok = check_can_block(dest, move_rsv, stop_rsv);
          // cout << "# c = " << c <<  " pets = " << pets << " isok = " << isok << "\n";
          if (!pets.empty() && isok) {
            act[i] = c;
            reserve_stop(i, c, stop_rsv);
            // for (auto val:pets) ps[val] = true;
          }
        }
      }
    }

    if (human[i].meta < 0 || act[i] != ACT_NONE || hp[i] == human[i].subject_pos) continue;

    // char c = decide_move_dir(i, d, 0, true);
    // // cout << "# i=" << i+1 << ", c=" << c << ", pnp=" << pnp[human[i].meta] << ", subject_pos=" << human[i].subject_pos << "\n";
    // act[i] = c;
    // reserve_move(i, c, move_rsv);
  }

  rep(i, M) {
    if (human[i].job != 53) continue;

    rep(k, 4) {
      int t = k;
      if (human[i].meta >= 2) t = (t+2)%4;
      if (human[i].meta == t) continue;

      char c = ACT_STOP[t];
      P p = dest_pos(hp[i], c);
      if (!blocked[p.first][p.second]) rem[i]++;
      if (check_can_block(p, move_rsv, stop_rsv)) cnt[i]++;
    }
    if (rem[i] != cnt[i] || rem[i] > 1) is_last_one = false;
  }

  rep(i, M) {
    if (human[i].job != 53) continue;
    // cout << "# " << i+1 << " : meta=" << human[i].meta << "\n";

    rep(k, 4) {
      int t = k;
      if (human[i].meta < 2) t = (t+2)%4;
      if (human[i].meta == t) continue;

      char c = ACT_STOP[t];
      P p = dest_pos(hp[i], c);
      if (check_can_block(p, move_rsv, stop_rsv)) {
        if (rem[i] == 1 && !is_last_one) continue;
        act[i] = c;
        reserve_stop(i, c, stop_rsv);
        break;
      }
    }
  }

}

void final_operation()
{
  if (phase == PHASE_8) {
    final_operation_asign();
  }

  if (phase == PHASE_9) {
    final_operation_prepare();
    final_operation_action();
  }

}

void plan_b()
{
  rep(i, M) act[i] = ACT_NONE;

  // 初期人員割当て
  if (phase == PHASE_0) {
    b_phase_1_asign();
    phase = PHASE_1;
  }

  // 地形作成
  if (phase == PHASE_1) {
    b_phase_1_build();
  }

}

/**
 * @brief 最初の入力受け取り
 */
void input()
{
  cin >> N;
  pet_remains = N;
  vector<int> tmp(5);
  rep(i, N) {
    int x, y, t;
    cin >> x >> y >> t;
    x--; y--; t--;

    pp[i] = {x, y};
    pt[i] = t;
    rep(j, 4) {
      int nx = x + dx[j], ny = y + dy[j];
      if (nx < 0 || H <= nx || ny < 0 || W <= ny) continue;
    }
    pf[x][y] = 1<<i;
    if (pt[i] == DOG) dog_exists = true;
    tmp[pt[i]]++;
  }
  if (!dog_exists) debug("there are no dogs.");
  else {
    if (tmp[DOG] <= FEW_DOGS_THRESHOLD) {
      swap(base, base_few_dogs);
      swap(build_sequence_center, build_sequence_center_few_dogs);
      swap(phase_1_startpos, phase_1_startpos_few_dogs);
      swap(phase_1_waitpos, phase_1_waitpos_few_dogs);
    }
  }
  debug(tmp);

  cin >> M;
  rep(i, M) {
    int x, y;
    cin >> x >> y;
    x--; y--;

    hp[i] = {x, y};
    hf[x][y] = 1<<i;

    act.push_back('.');
  }

}

/**
 * @brief デバッグ用出力
 */
void debug_output()
{
  // cout << "# turn : " << 300-turn << "\n";
  // cout << "# phase = " << phase << "\n";

  // rep(i, M) {
  //   cout << "# human " << i+1 << " : job=" << human[i].job << " now=" << hp[i] << " subject_pos=" << human[i].subject_pos << " meta=" << human[i].meta << " base=" << base[hp[i].first][hp[i].second] << "\n";
  //   if (human[i].job == 42 && human[i].meta >= 0) {
  //     cout << "# target number=" << human[i].meta+1 << ", pnp = " << pnp[human[i].meta] << "\n";
  //   }
  // }

  // rep(i, N) {
  //   cout << "# pet " << i+1 << (ps[i]?" 1":" 0") << " pnp=" << pnp[i] << "\n";
  // }

}

/**
 * @brief 出力
 */
void output()
{
  humans_action();
  update_nearest_passage_from_pets();

  debug_output();

  cout << act << "\n";
  pets_action();
}

/**
 * @brief 全員をランダムに動かす
 *
 */
void move_random()
{
  rep(i, M) {
    char c = ACT_MOVE[get_rand_range(0, 3)];
    P p = dest_pos(hp[i], c);
    if (p.first < 0 || H <= p.first || p.second < 0 || W <= p.second) {
      i--; continue;
    }
    act[i] = c;
  }
  output();
  turn--;
}

/**
 * @brief メイン関数
 *
 * @return int
 */
int main()
{
  srand((unsigned)time(NULL));
  int ti = clock();

  input();

  if (!dog_exists) {
    swap(base, base_nodog);
  }

  init();

  move_random();

  if (dog_exists) {
    while (turn--) {
      plan_a();
      if (phase < PHASE_8) catching_pets_operation();
      else final_operation();
      output();
    }

  } else {
    while (turn--) {
      plan_b();
      if (phase < PHASE_8) catching_pets_operation();
      else final_operation();
      output();
    }

  }
  int tim = (clock() - ti) / 1000;
  debug(tim);

  return 0;
}
