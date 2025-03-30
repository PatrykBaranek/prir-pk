import matplotlib.pyplot as plt
import numpy as np

methods = ['Sequential', 'Inner Parallel', 'Outer Parallel']
times = [5.306792, 2.458152, 1.598719]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(methods, times, color=['#3498db', '#2ecc71', '#e74c3c'], width=0.6)

ax.set_title('Matrix Multiplication Performance Comparison', fontsize=16)
ax.set_xlabel('Implementation Method', fontsize=14)
ax.set_ylabel('Execution Time (seconds)', fontsize=14)

ax.set_ylim(bottom=0)

ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{height:.2f}s', ha='center', fontsize=12)

sequential_time = times[0]
for i, time in enumerate(times[1:], 1):
    speedup = (sequential_time - time) / sequential_time * 100
    ax.text(i, times[i] / 2, f'Speedup: {speedup:.1f}%', 
            ha='center', color='white', fontweight='bold')

plt.tight_layout()

plt.savefig('./plots/7/matrix_comparison.png', dpi=300)