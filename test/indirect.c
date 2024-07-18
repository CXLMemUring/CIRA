#include <stdio.h>
#include <stdlib.h>
#define N 100000000
int a[N];
int b[N];
int main() {
    int c = 0;
    for (int i = 0; i < N; i++) {
        a[i] = rand() % N;
        b[i] = rand() % N;
    }
    for (int i = 0; i < N; i++) {
        c += a[b[i++]];
    }
    printf("%d\n", c);
}