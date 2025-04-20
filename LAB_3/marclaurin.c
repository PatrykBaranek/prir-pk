#include <stdio.h>
#include <math.h>
#include <time.h>

double factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

double sin_maclaurin(double x, int N) {
    double sum = 0.0;
    for (int i = 0; i <= N; i++) {
        int sign = (i % 2 == 0) ? 1 : -1;
        double term = sign * pow(x, 2 * i + 1) / factorial(2 * i + 1);
        sum += term;
    }
    return sum;
}

int main() {
    double x;
    int N;
    printf("Podaj x (w radianach): ");
    scanf("%lf", &x);
    printf("Podaj liczbe wyrazow szeregu N: ");
    scanf("%d", &N);

    clock_t start = clock();
    double approx = sin_maclaurin(x, N);
    clock_t end = clock();

    double exact = sin(x);
    double abs_error = fabs(approx - exact);

    printf("Sekwencyjnie:\n");
    printf("sin(x) z szeregu:           %.15f\n", approx);
    printf("sin(x) z biblioteki math.h: %.15f\n", exact);
    printf("Blad bezwzgledny:           %.15e\n", abs_error);
    printf("Czas wykonania:             %.6f s\n\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
