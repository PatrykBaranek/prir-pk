#include <time.h>
#include <stdio.h>
#include <math.h>

double f(double x) {
	return 1 / (1 + x * x);
}

double trapezoidal_cpu(double a, double b, int n) {
	double h = (b - a) / n;
	double sum = 0.0;

	sum += f(a) / 2.0;
	sum += f(b) / 2.0;

	for (int i = 1; i < n; i++) {
		double x = a + i * h;
		sum += f(x);
	}
	return (b - a) * sum / n;
}

int main() {
	double a = 0;
	double b = 1;
	int n = 10000;

	clock_t start = clock();
	double result = trapezoidal_cpu(a, b, n);
	clock_t end = clock();

	double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;

	printf("CPU: wynik = %.10f, czas = %.6f s\n", 4.0 * result, cpu_time);

	return 0;
}