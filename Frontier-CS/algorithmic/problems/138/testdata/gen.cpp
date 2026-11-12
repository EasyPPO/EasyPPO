#include <bits/stdc++.h>
using namespace std;

static const string CH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

char rndCh() {
    return CH[rng() % CH.size()];
}

struct Preset {
    int r, c;
    vector<string> mat;
};

int main() {
    const int T = 10;
    const int n = 20, m = 20, k = 20;

    for (int tc = 1; tc <= T; ++tc) {
        // 1. Random target
        vector<string> target(n, string(m, 'A'));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                target[i][j] = rndCh();

        // 2. Random presets
        vector<Preset> pres(k + 1);
        for (int p = 1; p <= k; ++p) {
            int pr = uniform_int_distribution<int>(1, n)(rng);
            int pc = uniform_int_distribution<int>(1, m)(rng);
            pres[p].r = pr;
            pres[p].c = pc;
            pres[p].mat.assign(pr, string(pc, 'A'));
            for (int i = 0; i < pr; ++i)
                for (int j = 0; j < pc; ++j)
                    pres[p].mat[i][j] = rndCh();
        }

        // 3. Generate random operation sequence (forward), apply in reverse
        //    Constraints: total ops <= 4e5, preset <= 400
        //    For randomness, length can be random between 2e5~4e5
        int maxOps = 400000;
        int wantOps = uniform_int_distribution<int>(200000, 400000)(rng);

        struct Op { int op, x, y; };
        vector<Op> ops;
        ops.reserve(wantOps);

        int presetUsed = 0;

        // Start from target and work backwards
        vector<string> cur = target;

        auto doBackwardRotate = [&](int x, int y) {
            // Reverse: undo clockwise rotation, equivalent to counter-clockwise
            // indices: (x,y), (x,y+1), (x+1,y+1), (x+1,y)
            // Original clockwise: new = {d, a, b, c}
            // Reverse: old = {b, c, d, a}
            char ny  = cur[x][y+1];
            char nyy = cur[x+1][y+1];
            char nyyy= cur[x+1][y];
            char nyyyy= cur[x][y];
            // Restore old
            cur[x][y]     = ny;
            cur[x][y+1]   = nyy;
            cur[x+1][y+1] = nyyy;
            cur[x+1][y]   = nyyyy;
        };

        for (int i = 0; i < wantOps; ++i) {
            // Choose a random operation for variety
            // 0:-4,1:-3,2:-2,3:-1,4:rotate(0),5:preset
            int typ = uniform_int_distribution<int>(0, 5)(rng);
            int op, x, y;

            if (typ == 5 && presetUsed >= 400) {
                typ = uniform_int_distribution<int>(0, 4)(rng);
            }

            if (typ == 0) { // -4: swap with up
                if (n <= 1) { --i; continue; }
                x = uniform_int_distribution<int>(2, n)(rng);
                y = uniform_int_distribution<int>(1, m)(rng);
                op = -4;
                // backward: same swap
                swap(cur[x-1][y-1], cur[x-2][y-1]);
            } else if (typ == 1) { // -3: swap with down
                if (n <= 1) { --i; continue; }
                x = uniform_int_distribution<int>(1, n-1)(rng);
                y = uniform_int_distribution<int>(1, m)(rng);
                op = -3;
                swap(cur[x-1][y-1], cur[x][y-1]);
            } else if (typ == 2) { // -2: swap with left
                if (m <= 1) { --i; continue; }
                x = uniform_int_distribution<int>(1, n)(rng);
                y = uniform_int_distribution<int>(2, m)(rng);
                op = -2;
                swap(cur[x-1][y-1], cur[x-1][y-2]);
            } else if (typ == 3) { // -1: swap with right
                if (m <= 1) { --i; continue; }
                x = uniform_int_distribution<int>(1, n)(rng);
                y = uniform_int_distribution<int>(1, m-1)(rng);
                op = -1;
                swap(cur[x-1][y-1], cur[x-1][y]);
            } else if (typ == 4) { // rotate
                if (n <= 1 || m <= 1) { --i; continue; }
                x = uniform_int_distribution<int>(1, n-1)(rng);
                y = uniform_int_distribution<int>(1, m-1)(rng);
                op = 0;
                // backward rotate
                doBackwardRotate(x-1, y-1);
            } else {
                // preset
                int pid = uniform_int_distribution<int>(1, k)(rng);
                int pr = pres[pid].r;
                int pc = pres[pid].c;
                x = uniform_int_distribution<int>(1, n - pr + 1)(rng);
                y = uniform_int_distribution<int>(1, m - pc + 1)(rng);
                op = pid;
                presetUsed++;

                // backward: fill this region with random chars
                for (int ii = 0; ii < pr; ++ii)
                    for (int jj = 0; jj < pc; ++jj)
                        cur[x - 1 + ii][y - 1 + jj] = rndCh();
            }

            ops.push_back({op, x, y});
        }

        // Now cur is the initial state
        vector<string> initial = cur;

        // 4. Write to file
        {
            string fname = to_string(tc) + ".in";
            ofstream fout(fname);
            fout << n << " " << m << " " << k << "\n";
            for (int i = 0; i < n; ++i) fout << initial[i] << "\n";
            fout << "\n";
            for (int i = 0; i < n; ++i) fout << target[i] << "\n";
            for (int p = 1; p <= k; ++p) {
                fout << "\n";
                fout << pres[p].r << " " << pres[p].c << "\n";
                for (int i = 0; i < pres[p].r; ++i)
                    fout << pres[p].mat[i] << "\n";
            }
            fout.close();
        }

        // 5. Output empty ans file
        {
            string fname = to_string(tc) + ".ans";
            ofstream fout(fname);
            // empty
            fout.close();
        }

        cerr << "generated " << tc << ".in/.out, ops=" << ops.size()
             << ", presetsUsed=" << presetUsed << "\n";
    }

    return 0;
}
