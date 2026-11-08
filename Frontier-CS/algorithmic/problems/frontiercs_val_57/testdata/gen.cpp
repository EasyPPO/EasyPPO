#include <bits/stdc++.h>
using namespace std;

struct Edge{int u,v;};

// ---------- RNG ----------
static uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static uint64_t seed_from_clock(){
    uint64_t t = chrono::high_resolution_clock::now().time_since_epoch().count();
    return splitmix64(t) ^ splitmix64(t<<1);
}
struct RNG {
    using ull = unsigned long long;
    mt19937_64 eng;
    RNG(ull s):eng(s){}
    int randint(int l,int r){ uniform_int_distribution<int> d(l,r); return d(eng); }
    double rand01(){ uniform_real_distribution<double> d(0.0,1.0); return d(eng); }
    template<class It> void shuffle(It b, It e){ std::shuffle(b,e,eng); }
};

// ---------- relabel ----------
vector<Edge> relabel(int n, const vector<Edge>& E, RNG& rng, int& root){
    vector<int> p(n+1); iota(p.begin(), p.end(), 0);
    if(n >= 2) rng.shuffle(p.begin()+1, p.end());
    vector<Edge> R; R.reserve((size_t)max(0,n-1));
    for(auto e:E) R.push_back({p[e.u], p[e.v]});
    root = p[root];
    return R;
}

// ---------- generators ----------
vector<Edge> gen_path(int n, int& root, RNG& rng){
    (void)rng; root=1; vector<Edge> E; E.reserve(n-1);
    for(int i=2;i<=n;i++) E.push_back({i-1,i});
    return E;
}
vector<Edge> gen_star(int n, int& root, RNG& rng){
    int c=rng.randint(1,n); root=c; vector<Edge> E; E.reserve(n-1);
    for(int v=1; v<=n; v++) if(v!=c) E.push_back({c,v});
    return E;
}
vector<Edge> gen_broom(int n, int& root, RNG& rng){
    int L=max(2, rng.randint(n/5, max(2,n*3/5)));
    vector<Edge> E; E.reserve(n-1);
    for(int i=2;i<=L;i++) E.push_back({i-1,i});
    int hub=L; root=hub;
    for(int v=L+1; v<=n; v++) E.push_back({hub,v});
    return E;
}
vector<Edge> gen_double_star(int n, int& root, RNG& rng){
    (void)rng; root=1; vector<Edge> E; E.reserve(n-1);
    E.push_back({1,2});
    for(int v=3; v<=n; v++) E.push_back({ (v&1)?1:2, v});
    return E;
}
vector<Edge> gen_caterpillar(int n, int& root, RNG& rng){
    int L = rng.randint(max(3,(int)round(pow(n,0.5))), min(n-1, max(3,(int)round(pow(n,0.85)))));
    vector<Edge> E; E.reserve(n-1);
    for(int i=2;i<=L;i++) E.push_back({i-1,i});
    for(int v=L+1; v<=n; v++) E.push_back({rng.randint(1,L), v});
    root = rng.randint(1,L);
    return E;
}
vector<Edge> gen_balanced_kary(int n, int& root, RNG& rng){
    int k=rng.randint(2,10); root=1; vector<Edge> E; E.reserve(n-1);
    queue<int> q; q.push(1); int cur=2;
    while(cur<=n){
        int u=q.front(); q.pop();
        int c=min(k, n-cur+1);
        for(int i=0;i<c;i++){ int v=cur++; E.push_back({u,v}); q.push(v); }
    }
    return E;
}
vector<Edge> gen_binary_heap(int n, int& root, RNG& rng){
    (void)rng; root=1; vector<Edge> E; E.reserve(n-1);
    for(int v=2; v<=n; v++) E.push_back({v/2, v});
    return E;
}
vector<Edge> gen_recursive_tree(int n, int& root, RNG& rng){
    root=1; vector<Edge> E; E.reserve(n-1);
    for(int v=2; v<=n; v++) E.push_back({rng.randint(1,v-1), v});
    return E;
}
vector<Edge> gen_preferential_attachment(int n, int& root, RNG& rng){
    root=1; vector<Edge> E; E.reserve(n-1);
    vector<int> bag; bag.reserve(2*n);
    auto add=[&](int v,int t){ while(t--) bag.push_back(v); };
    E.push_back({1,2}); add(1,1); add(2,1);
    for(int v=3; v<=n; v++){ int u=bag[rng.randint(0,(int)bag.size()-1)]; E.push_back({u,v}); add(u,1); add(v,1); }
    return E;
}
vector<Edge> gen_uniform_prufer(int n, int& root, RNG& rng){
    root=1;
    vector<int> pr(max(0,n-2)); for(int i=0;i<(int)pr.size();i++) pr[i]=rng.randint(1,n);
    vector<int> deg(n+1,1); for(int x:pr) deg[x]++;
    priority_queue<int, vector<int>, greater<int>> pq;
    for(int i=1;i<=n;i++) if(deg[i]==1) pq.push(i);
    vector<Edge> E; E.reserve(n-1);
    for(int x:pr){
        int u=pq.top(); pq.pop();
        E.push_back({u,x});
        if(--deg[x]==1) pq.push(x);
    }
    if(n>=2){ int u=pq.top(); pq.pop(); int v=pq.top(); pq.pop(); E.push_back({u,v}); }
    return E;
}

// ---------- partition SUM_N into T positive ints ----------
vector<int> random_partition_sum(int S, int T, RNG& rng){
    // assumes 1 <= T <= S
    vector<int> cuts; cuts.reserve(T-1);
    unordered_set<int> used;
    while((int)cuts.size() < T-1){
        int x = rng.randint(1, S-1);
        if(!used.count(x)){ used.insert(x); cuts.push_back(x); }
    }
    sort(cuts.begin(), cuts.end());
    vector<int> parts; parts.reserve(T);
    int prev=0;
    for(int c: cuts){ parts.push_back(c - prev); prev = c; }
    parts.push_back(S - prev);
    rng.shuffle(parts.begin(), parts.end());
    return parts;
}

int main(int argc, char** argv){
    freopen("15.in","w",stdout);
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Adjustable parameters: seed, T, SUM_N
    uint64_t seed = seed_from_clock();
    int T = 1;
    int SUM_N = 1000;

    if(argc >= 2){ string s=argv[1]; stringstream ss(s); ss>>seed; if(!ss) seed=seed_from_clock(); }
    if(argc >= 3){ string s=argv[2]; stringstream ss(s); ss>>T; }
    if(argc >= 4){ string s=argv[3]; stringstream ss(s); ss>>SUM_N; }

    if(T <= 0){ cerr << "Error: T must be positive.\n"; return 1; }
    if(SUM_N < T){ cerr << "Error: SUM_N must be >= T (each tree size >=1).\n"; return 1; }

    RNG rng(seed);

    using GenFn = function<vector<Edge>(int,int&,RNG&)>;
    vector<GenFn> base = {
        gen_uniform_prufer,
        gen_path,
        gen_star,
        gen_broom,
        gen_double_star,
        gen_caterpillar,
        gen_balanced_kary,
        gen_binary_heap,
        gen_recursive_tree,
        gen_preferential_attachment
    };
    rng.shuffle(base.begin(), base.end());

    // Build generator plan of length >= T: repeat shuffled base until sufficient
    vector<GenFn> gens;
    while((int)gens.size() < T){
        for(auto &g: base) gens.push_back(g);
    }
    gens.resize(T); // Exactly T generators

    // Randomly partition SUM_N into T parts
    vector<int> sizes = random_partition_sum(SUM_N, T, rng);

    cout << T << "\n";
    for(int i=0;i<T;i++){
        int n = sizes[i];
        int root = 1;
        vector<Edge> E = gens[i](n, root, rng);
        int new_root = root;
        vector<Edge> ER = relabel(n, E, rng, new_root);

        cout << n << "\n";
        for(auto &e: ER) cout << e.u << " " << e.v << "\n";
        cout << new_root << "\n";
    }
    return 0;
}
