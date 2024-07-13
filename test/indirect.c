#include <stdlib.h>
#define N 1000000
int main() {
    int a[N];
    int b[N];
    int c;
    for (int i = 0; i < N; i++) {
        a[i] = rand();
        b[i] = rand();
    }
    for (int i = 0; i < N; i++) {
        c += a[b[i++]];
    }
}