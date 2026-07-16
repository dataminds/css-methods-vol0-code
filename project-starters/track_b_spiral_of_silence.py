"""트랙 B · 침묵의 나선 — 기말 프로젝트 시작 코드 (self-contained, Mesa 3.x).

요지: 자기가 소수라고 지각하면 표명을 자제한다. 표명된 것이 다시 지각을 이룬다.
ODD-lite: 목적=슬쩍 다수가 표명을 지배하는가 / 행위자=시민 / 상태=사적의견(고정)+표명여부 /
  규칙: 표명자 표본에서 내 편 비율 지각 → 소수면 침묵 확률 / 시간=매 스텝 전원 /
  출력=표명자 중 의견1 비율 / 한계=잘 섞인 표본(네트워크 없음).
서명: 실제 다수(예 0.55)가 표명에서 더 크게 우세해진다.

실행: python track_b_spiral_of_silence.py
"""
from __future__ import annotations
import numpy as np
import mesa


class Citizen(mesa.Agent):
    def __init__(self, model, opinion):
        super().__init__(model)
        self.opinion = opinion
        self.expressing = True

    def step(self):
        m = self.model
        others = [a for a in m.agents if a is not self and a.expressing]
        if not others:
            self.expressing = True
            return
        sample = m.random.sample(others, min(m.k, len(others)))
        perceived = sum(1 for a in sample if a.opinion == self.opinion) / len(sample)
        self.expressing = True if perceived >= m.threshold else (m.random.random() > m.silence_prob)


class SpiralOfSilence(mesa.Model):
    def __init__(self, n=120, p1=0.55, k=8, threshold=0.5, silence_prob=0.7, seed=None):
        super().__init__(seed=seed)
        self.k, self.threshold, self.silence_prob = k, threshold, silence_prob
        for _ in range(n):
            Citizen(self, 1 if self.random.random() < p1 else 0)


def expressed_share_1(model):
    expr = [a.opinion for a in model.agents if a.expressing]
    return float(np.mean(expr)) if expr else float("nan")


if __name__ == "__main__":
    m = SpiralOfSilence(p1=0.55, seed=1)
    print(f"실제 의견1 비율 0.55 / 표명 의견1 시작 {expressed_share_1(m):.2f}")
    for _ in range(30):
        m.agents.shuffle_do("step")
    print(f"표명 의견1 끝 {expressed_share_1(m):.2f}  (더 커지면 나선)")

# 확장 아이디어(TODO): 네트워크(이웃만 지각) / 완고한 소수(hard-core) / 이슈 여러 개 /
#   미디어가 다수 의견을 증폭 / 지각 오차(잘못 지각).
