#include <stdio.h>
int remote(int a, int b[], int c[]) {
    int res = 0;
    for (int i = a; i < a + 4; i++) {
        res += b[c[b[i]]];
    }
    return res;
}