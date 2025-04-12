import matplotlib.pyplot as plt

processes = [1, 2, 3, 4]
times = [0.000010, 0.000051, 0.000105, 0.000099]

plt.figure(figsize=(8, 6))
plt.plot(processes, times, marker='o', linestyle='-', color='blue', label='Czas obliczeń')

plt.title('Zależność czasu obliczeń od liczby procesów', fontsize=14)
plt.xlabel('Liczba procesów', fontsize=12)
plt.ylabel('Czas obliczeń (sekundy)', fontsize=12)
plt.xticks(processes)
plt.grid(True)
plt.legend()
plt.savefig('./plots/pi_plots.png')