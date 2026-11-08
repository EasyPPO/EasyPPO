#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pr pair<int, int>
#define pb push_back
#define mid (l + r) / 2
#define ls num << 1
#define rs num << 1 | 1

inline int read() {
    int x = 0, m = 1;
    char ch = getchar();

    while (!isdigit(ch)) {
        if (ch == '-') m = -1;
        ch = getchar();
    }

    while (isdigit(ch)) {
        x = x * 10 + ch - 48;
        ch = getchar();
    }

    return x * m;
}

inline void write(int x) {
    if (x < 0) {
        putchar('-');
        write(-x);
        return;
    }

    if (x >= 10) write(x / 10);
    putchar(x % 10 + '0');
}

#define Pr pair<int, pr>

const int N = 25;

char a[N][N], b[N][N], c[N][N][N], d[N][N];
int X[N], Y[N], Cnt[N][128], vis[N][N];
int cnt[128];

vector<Pr> ans;

char get() {
    char ch = getchar();
    while ((ch < 'a' || ch > 'z') && (ch < 'A' || ch > 'Z') && !isdigit(ch)) ch = getchar();
    return ch;
}

signed main() {
    // freopen("2.in","r",stdin);
    // freopen("1.ans","w",stdout);
    int n = read(), m = read(), k = read();
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            a[i][j] = get();
        }
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            b[i][j] = get();
            cnt[b[i][j]]++;
        }
    }
    for (int x = 1; x <= k; x++) {
        int n1 = read(), m1 = read();
        X[x] = n1;
        Y[x] = m1;
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= m1; j++) {
                c[x][i][j] = get();
                Cnt[x][c[x][i][j]]++;
            }
        }
    }
    while (1) {
        int pos = 0;
        for (int i = 1; i <= k; i++) {
            int res = 0, p = 0;
            for (int j = 0; j < 128; j++) {
                res += max(0ll, Cnt[i][j] - cnt[j]);
                if (Cnt[i][j] && cnt[j]) p = 1;
            }
            if (res <= cnt[32] && p) {
                pos = i;
                break;
            }
        }
        if (!pos) break;
        memset(vis, 0, sizeof(vis));
        int n1 = X[pos], m1 = Y[pos];
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= m1; j++) {
                int xx = 0, yy = 0;
                for (int x = 1; x <= n; x++) {
                    for (int y = 1; y <= m; y++) {
                        if (vis[x][y]) continue;
                        if (b[x][y] == c[pos][i][j]) {
                            xx = x;
                            yy = y;
                            break;
                        }
                    }
                    if (xx) break;
                }
                if (!xx) {
                    for (int x = 1; x <= n; x++) {
                        for (int y = 1; y <= m; y++) {
                            if (vis[x][y]) continue;
                            if (x == i && y < j) continue;
                            if (b[x][y] == ' ') {
                                xx = x;
                                yy = y;
                                break;
                            }
                        }
                        if (xx) break;
                    }
                }
                while (xx < i) {
                    ans.pb({-3, {xx, yy}});
                    swap(b[xx][yy], b[xx + 1][yy]);
                    xx++;
                }
                while (yy < j) {
                    ans.pb({-1, {xx, yy}});
                    swap(b[xx][yy], b[xx][yy + 1]);
                    yy++;
                }
                while (yy > j) {
                    ans.pb({-2, {xx, yy}});
                    swap(b[xx][yy], b[xx][yy - 1]);
                    yy--;
                }
                while (xx > i) {
                    ans.pb({-4, {xx, yy}});
                    swap(b[xx][yy], b[xx - 1][yy]);
                    xx--;
                }
                vis[i][j] = 1;
                cnt[b[i][j]]--;
                b[i][j] = ' ';
                cnt[b[i][j]]++;
            }
        }
        ans.pb({pos, {1, 1}});
        while (1) {
            int xx = 0, yy = 0;
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= m; j++) {
                    if (vis[i][j]) continue;
                    if (Cnt[pos][b[i][j]]) {
                        xx = i;
                        yy = j;
                        break;
                    }
                }
                if (xx) break;
            }
            if (!xx) break;
            int p = 0;
            for (int i = 1; i <= n1; i++) {
                for (int j = 1; j <= m1; j++) {
                    if (c[pos][i][j] == b[xx][yy]) {
                        p = 1;
                        while (xx < i) {
                            ans.pb({-3, {xx, yy}});
                            swap(b[xx][yy], b[xx + 1][yy]);
                            xx++;
                        }
                        while (yy < j) {
                            ans.pb({-1, {xx, yy}});
                            swap(b[xx][yy], b[xx][yy + 1]);
                            yy++;
                        }
                        while (yy > j) {
                            ans.pb({-2, {xx, yy}});
                            swap(b[xx][yy], b[xx][yy - 1]);
                            yy--;
                        }
                        while (xx > i) {
                            ans.pb({-4, {xx, yy}});
                            swap(b[xx][yy], b[xx - 1][yy]);
                            xx--;
                        }
                        cnt[b[i][j]]--;
                        b[i][j] = ' ';
                        cnt[b[i][j]]++;
                        ans.pb({pos, {1, 1}});
                        break;
                    }
                }
                if (p) break;
            }
        }
    }
    memset(vis, 0, sizeof(vis));
    int P = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int xx = 0, yy = 0;
            for (int x = 1; x <= n; x++) {
                for (int y = 1; y <= m; y++) {
                    if (vis[x][y]) continue;
                    if (b[x][y] == a[i][j]) {
                        xx = x;
                        yy = y;
                        break;
                    }
                }
                if (xx) break;
            }
            if (!xx) {
                for (int x = 1; x <= n; x++) {
                    for (int y = 1; y <= m; y++) {
                        if (vis[x][y]) continue;
                        if (b[x][y] == ' ') {
                            xx = x;
                            yy = y;
                            break;
                        }
                    }
                    if (xx) break;
                }
            }
            if (!xx) {
                P = 0;
                break;
            }
            while (xx < i) {
                ans.pb({-3, {xx, yy}});
                swap(b[xx][yy], b[xx + 1][yy]);
                xx++;
            }
            while (yy < j) {
                ans.pb({-1, {xx, yy}});
                swap(b[xx][yy], b[xx][yy + 1]);
                yy++;
            }
            while (yy > j) {
                ans.pb({-2, {xx, yy}});
                swap(b[xx][yy], b[xx][yy - 1]);
                yy--;
            }
            while (xx > i) {
                ans.pb({-4, {xx, yy}});
                swap(b[xx][yy], b[xx - 1][yy]);
                xx--;
            }
            vis[i][j] = 1;
        }
        if (!P) break;
    }
    if (!P) {
        puts("-1");
        return 0;
    }
    reverse(ans.begin(), ans.end());
    write(ans.size());
    putchar('\n');
    for (auto u : ans) {
        write(u.first);
        putchar(' ');
        write(u.second.first);
        putchar(' ');
        write(u.second.second);
        putchar('\n');
    }
}