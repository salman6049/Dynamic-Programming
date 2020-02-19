#include <cstdio>
#include <vector>
using namespace std;

#define REP(i, n) for (decltype(n) i = 0; i < (n); i++)
#define ROF(i, a, b) for (decltype(b) i = (b); --i >= (a); )

int ri()
{
  int x;
  scanf("%d", &x);
  return x;
}

const int N = 50000;
int a[N], dep[N], pre[N], post[N], nim[N][2], tick;
vector<int> es[N];

void dfs(int u, int d, int p)
{
  nim[u][d] += a[u];
  dep[u] = d;
  pre[u] = tick++;
  for (int v: es[u])
    if (v != p) {
      dfs(v, d^1, u);
      REP(i, 2)
        nim[u][i] ^= nim[v][i];
    }
  post[u] = tick;
}

int main()
{
  int n = ri();
  REP(i, n)
    a[i] = ri();
  REP(i, n-1) {
    int u = ri()-1, v = ri()-1;
    es[u].push_back(v);
    es[v].push_back(u);
  }
  dfs(0, 0, -1);
  ROF(_, 0, ri()) {
    int u = ri()-1, v = ri()-1;
    if (pre[u] <= pre[v] && pre[v] < post[u])
      puts("INVALID");
    else
      puts(nim[0][1] ^ (dep[u] ^ dep[v] ? 0 : nim[u][0] ^ nim[u][1]) ? "YES" : "NO");
  }
}
