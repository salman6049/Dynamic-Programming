import java.io.PrintWriter;
import java.util.Scanner;

public class Solution{
    public static final int MOD = (int) 1e9 + 7;
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        while(sc.hasNext()){
            solve(sc, pw);
        }
        sc.close();
        pw.flush();
        pw.close();
    }

    private static void solve(Scanner sc, PrintWriter pw){
        int N = sc.nextInt();
        int K = sc.nextInt();
        K = N - K;

        int[][] stains = new int[N+1][2];
        int[] vals = new int[]{0,100000,0,100000};
        for(int i = 1; i <= N; i++){
            stains[i][0] = sc.nextInt();
            stains[i][1] = sc.nextInt();
            vals[0] = Math.max(vals[0], stains[i][0]);
            vals[1] = Math.min(vals[1], stains[i][0]);
            vals[2] = Math.max(vals[2], stains[i][1]);
            vals[3] = Math.min(vals[3], stains[i][1]);
        }

        if(K == 0){
            pw.println(1);
            return;
        }

        int[] arr = new int[N+1];
        for(int i = 1; i <= N; i++) {
            int mask = 0;
            for(int j = 0; j < 4; j++){
                if(vals[j] == stains[i][j/2]){
                    mask |= (1 << j);
                }
            }
            arr[i] = mask;
        }

        int[][][] dp = new int[K+1][N+1][16];

        for(int j = 1; j <= N; j++){
            dp[1][j][arr[j]] = 1;
            for(int k = 0; k < 16; k++){
                dp[1][j][k] += dp[1][j-1][k];
            }
        }

        for(int i = 1; i < K; i++){
            for(int j = i; j < N; j++){
                for(int k = 0; k < 16; k++){
                    dp[i+1][j+1][k | arr[j+1]] = (dp[i+1][j+1][k | arr[j+1]] + dp[i][j][k]) % MOD;
                    dp[i+1][j+1][k] = (dp[i+1][j+1][k] + dp[i+1][j][k]) % MOD;
                }
            }
        }

        int ans = 0;
        for(int k = 0; k < 15; k++){
            ans = (ans + dp[K][N][k]) % MOD;
        }
        pw.println(ans);
    }


}
