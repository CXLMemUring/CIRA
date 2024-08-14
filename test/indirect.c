#include <stdio.h>
#include <stdlib.h>
#define N 100
int a[N];
int b[N];
int main() {
    int c = 0;
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }
    for (int i = 0; i < N; i++) {
        c += a[b[a[i]]];
        // printf("b[%d]=%d\n", i, a[b[a[i]]]);
    }
    printf("c=%d\n",c);
    c=0;
    for (int d = 0; d < N; d += 4) {
        int res = 0;
        for (int i = d; i < d + 4; i++) {
            // printf("b[%d]=%d\n", a + i, b[d[b[a + i]]]);
            res += a[b[a[i]]];
        }
        c+=res;
        printf("d=%d\n", res);
    }
    printf("%d\n", c);
}