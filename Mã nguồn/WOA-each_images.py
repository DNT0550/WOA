import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hàm mục tiêu Sphere (2D)
def objective_function(x):
    return x[0]**2 + x[1]**2

# WOA thuật toán
def WOA_2D(max_iter, search_agents, lb, ub):
    whales = np.random.uniform(lb, ub, (search_agents, 2))
    leader = whales[0].copy()
    leader_score = objective_function(leader)

    positions_history = []
    convergence_curve = []

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(search_agents):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = random.random()
            b = 1
            l = random.uniform(-1, 1)

            if p < 0.5:
                if abs(A) >= 1:
                    rand_idx = random.randint(0, search_agents - 1)
                    D = abs(C * whales[rand_idx] - whales[i])
                    new_position = whales[rand_idx] - A * D
                else:
                    D = abs(C * leader - whales[i])
                    new_position = leader - A * D
            else:
                D = abs(leader - whales[i])
                new_position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + leader

            whales[i] = np.clip(new_position, lb, ub)
            score = objective_function(whales[i])
            if score < leader_score:
                leader_score = score
                leader = whales[i].copy()

        positions_history.append(whales.copy())
        convergence_curve.append(leader_score)

    return positions_history, convergence_curve


# Tạo animation và lưu thành GIF
def save_as_gif(positions_history, lb, ub, filename="woa_simulation.gif"):
    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame):
        ax.clear()
        positions = positions_history[frame]
        ax.set_xlim(lb-1, ub+1)
        ax.set_ylim(lb-1, ub+1)
        ax.set_title(f"WOA Simulation - Iteration {frame+1}")

        ax.scatter(positions[:,0], positions[:,1], s=100, label='Whales')

        best_idx = np.argmin([objective_function(p) for p in positions])
        leader_pos = positions[best_idx]
        ax.scatter(leader_pos[0], leader_pos[1], s=200, c='red', marker='*', label='Leader')

        ax.legend()

    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=400)
    anim.save(filename, writer='pillow')
    print(f"GIF saved as {filename}")

# Vẽ biểu đồ hội tụ
def plot_convergence(convergence_curve):
    plt.figure(figsize=(8,5))
    plt.plot(convergence_curve, linewidth=2)
    plt.title("WOA Convergence Curve", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Best Fitness Value", fontsize=12)
    plt.grid(True)
    plt.show()

# Thực thi
lb, ub = -10, 10
max_iter = 30
search_agents = 15

positions_history, convergence_curve = WOA_2D(max_iter, search_agents, lb, ub)

save_as_gif(positions_history, lb, ub, "woa_simulation.gif")
plot_convergence(convergence_curve)


