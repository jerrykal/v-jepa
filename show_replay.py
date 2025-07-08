import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 載入資料
data = np.load("test_1024.npz")
obs = data["obs"]         # shape: [T, N, H, W, C]
action = data["action"]   # shape: [T, N, A]
reward = data["reward"]   # shape: [T, N]
termination = data["done"]  # shape: [T, N]

env_id = 0  # 顯示哪個環境的資料
T = obs.shape[0]

# 建立圖形和子圖
fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))

# 顯示初始 obs
im = ax_img.imshow(obs[0, env_id])
ax_img.set_title("Observation")
text_action = ax_img.text(0.02, 0.95, '', transform=ax_img.transAxes,
                          color='white', fontsize=12, backgroundcolor='black')

# 初始 reward 曲線
reward_line, = ax_plot.plot([], [], label="Reward")
current_point, = ax_plot.plot([], [], 'ro')  # 紅點標記現在位置
ax_plot.set_xlim(0, T)
ax_plot.set_ylim(np.min(reward), np.max(reward))
ax_plot.set_title("Reward Over Time")
ax_plot.set_xlabel("Timestep")
ax_plot.set_ylabel("Reward")
ax_plot.legend()

# animation 更新函數
def update(frame):
    # 更新影像
    im.set_data(obs[frame, env_id])
    text_action.set_text(f"Action: {action[frame, env_id]}")

    # 更新 reward 曲線
    reward_so_far = reward[:frame+1, env_id]
    reward_line.set_data(np.arange(len(reward_so_far)), reward_so_far)

    # 更新目前 reward 的點
    current_reward = reward[frame, env_id]
    current_point.set_data([frame], [current_reward])

    # 若是 done，加上紅圈圈標記
    if termination[frame, env_id]:
        current_point.set_marker('o')
        current_point.set_markersize(10)
    else:
        current_point.set_marker('o')
        current_point.set_markersize(5)

    return im, reward_line, current_point, text_action

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
plt.tight_layout()
plt.show()
