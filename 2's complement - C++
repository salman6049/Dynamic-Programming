#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

int NumberOfSetBits(int i)
{
   i = i - ((i >> 1) & 0x55555555);
   i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
   return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

long long CountSetBit(int a)
{
 if(a == 0) return 0 ;
 if(a % 2 == 0) return CountSetBit(a - 1) + NumberOfSetBits(a) ;
 return ((long long)a + 1) / 2 + 2 * CountSetBit(a / 2) ;
}

long long CountSetBits(int a,int b)
{
 if(a >= 0)
 {
  long long ret = CountSetBit(b) ;
  if(a > 0) ret -= CountSetBit(a - 1) ;
  return ret ;
 }
 long long ret = (32LL * -(long long)a) - CountSetBit(~a) ;
 if(b > 0) ret += CountSetBit(b) ;
 else if(b < -1)
 {
  b++ ;
  ret -= (32LL * -(long long)b) - CountSetBit(~b) ;
 }
 return ret ;
}

int main()
{
 unsigned int T;
 long long int A,B ;
 scanf("%d", &T);
 while(T-- > 0)
 {
  scanf("%lld%lld", &A, &B);
  long long ret = CountSetBits(A,B) ;
  printf("%lld\n", ret);
 }
 return 0 ;
}
