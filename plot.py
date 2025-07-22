import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 불러오기
df = pd.read_csv("log.csv")

# 시간과 데이터 추출
time = df["time"]
r = df["r"]
force = df["force"]
ctrl = df["ctrl"]

# 플롯 생성
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 서브플롯 1: r
axs[0].plot(time.values, r.values, label="r", color="blue")
axs[0].set_ylabel("r")
axs[0].legend()
axs[0].grid(True)

# 서브플롯 2: force
axs[1].plot(time.values, force.values, label="force", color="red")
axs[1].set_ylabel("force")
axs[1].legend()
axs[1].grid(True)

# 서브플롯 3: ctrl
axs[2].plot(time.values, ctrl.values, label="ctrl", color="green")
axs[2].set_ylabel("ctrl")
axs[2].legend()
axs[2].grid(True)

# # 서브플롯 4: r vs ctrl 비교 (optional)
# axs[3].plot(time.values, r.values, label="r", color="blue", linestyle="--")
# axs[3].plot(time, ctrl, label="ctrl", color="green")
# axs[3].set_ylabel("r vs ctrl")
# axs[3].set_xlabel("time (s)")
# axs[3].legend()
# axs[3].grid(True)

plt.tight_layout()
plt.show()
