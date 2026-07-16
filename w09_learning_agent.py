"""9주 · 두꺼운 행위자 II: 상태 갱신(학습) = 예측오차 학습 (Mesa 3.x).

8주 행위자는 고정된 태도로 골랐다. 9주 행위자는 '경험으로 배운다'.
각 채널의 가치 추정 Q(상태)를 갖고, 받은 보상으로 갱신한다:
    Q[선택] += lr * (보상 - Q[선택])       (델타 규칙 = Rescorla-Wagner의 뼈대)
선택은 softmax(Q, tau).
서명: 좋은 채널을 고르는 비율이 시간에 따라 오른다. lr이 클수록 빨리 배운다.
절제: lr=0(학습 없음) → 좋은 채널 선택이 우연 수준(1/채널수)에 머문다.

실행: python w09_learning_agent.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import mesa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 같은 폴더 abmlib 임포트 보장
from abmlib import run, plot_series


def softmax(values, tau):
    v = np.asarray(values, dtype=float) / tau
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


class Learner(mesa.Agent):
    def __init__(self, model, k):
        super().__init__(model)
        self.Q = [0.0] * k          # 상태: 채널별 가치 추정
        self.chose_best = 0

    def step(self):
        m = self.model
        probs = softmax(self.Q, m.tau)                         # 선택 규칙
        c = m.random.choices(range(len(self.Q)), weights=probs)[0]
        reward = m.true_means[c] + m.random.gauss(0, m.noise)  # 환경이 준 보상(잡음 포함)
        self.Q[c] += m.lr * (reward - self.Q[c])               # 갱신: 예측오차 학습
        self.chose_best = int(c == m.best)


class BanditLearning(mesa.Model):
    def __init__(self, n=200, true_means=(0.2, 0.5, 0.8), tau=0.2, lr=0.3,
                 noise=0.2, seed=None):
        super().__init__(seed=seed)
        self.true_means = list(true_means)
        self.tau, self.lr, self.noise = tau, lr, noise
        self.best = int(np.argmax(true_means))
        k = len(true_means)
        for _ in range(n):
            Learner(self, k)
        self.datacollector = mesa.DataCollector(model_reporters={
            "best_choice_frac": lambda mm: float(np.mean([a.chose_best for a in mm.agents]))})

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    print("[서명] 학습이 있으면 좋은 채널 선택 비율이 오른다 (우연 = 1/3 = 약 0.33)")
    for lr in (0.0, 0.1, 0.4):
        df = run(BanditLearning(lr=lr, seed=1), 40)
        print(f"  lr={lr}: 좋은채널 선택 {df['best_choice_frac'].iloc[0]:.2f}"
              f" → {df['best_choice_frac'].iloc[-1]:.2f}")
    print("  → lr=0(절제)은 우연 수준에 머물고, lr이 클수록 빨리 좋은 채널로 수렴.")
    plot_series(run(BanditLearning(lr=0.3, seed=1), 40),
                "w09_learning.png", "value learning (lr=0.3)")
    print("  그림 저장: w09_learning.png")
