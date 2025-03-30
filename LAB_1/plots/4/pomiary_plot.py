import matplotlib.pyplot as plt

strategies = ['Static default', 'Static, chunk=3', 'Dynamic default', 'Dynamic, chunk=3']
times = [0.003049, 0.001715, 0.006260, 0.002928]

plt.figure(figsize=(10, 6))

colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
bars = plt.bar(strategies, times, color=colors, width=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
            f'{height:.6f}s', ha='center', va='bottom', fontsize=9)

plt.ylabel('Czas wykonania (sekundy)', fontweight='bold')
plt.xlabel('Strategia podziału', fontweight='bold')
plt.title('Porównanie strategii podziału pętli OpenMP', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.ylim(0, max(times) * 1.15)

plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
plt.tight_layout()

plt.axhline(y=min(times), color='green', linestyle='--', alpha=0.5)
plt.text(len(strategies)-1, min(times)*0.95, f'Best: {min(times):.6f}s', color='green', ha='right', fontsize=8)

plt.savefig('./4/pomiary_plot.png', dpi=300, bbox_inches='tight')