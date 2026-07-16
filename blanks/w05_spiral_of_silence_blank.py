# ===== 빈칸판(학생 배포용) =====
# TODO 블록을 채우고 raise 줄을 지우면, 완성본과 같은 서명이 나와야 합니다.
# 검증: 극단값 시험 + 씨앗 73 재실행 + (가능하면) 손 검산. 검증 로그 한 문단!

"""5주 · 소통 ABM: 침묵의 나선 (Noelle-Neumann) 최소판 (Mesa 3.x).

이론: 사람은 사적 의견을 갖지만, 주변에서 자기가 '소수'라고 지각하면 표명을 자제한다.
      표명된 의견이 다시 지각을 이루는 되먹임.
서명: 슬쩍 다수인 의견이 '표명'에서 나선형으로 우세해진다(표명 다수 > 실제 다수).
반증: 표명 다수가 실제 의견 다수(초기 비율)와 같은 수준에 머문다.

실행: python w05_spiral_of_silence.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import mesa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 상위 폴더(materials) abmlib 임포트
from abmlib import run, plot_series


def expressed_share_1(model):
    """'표명 중'인 사람들 사이에서 의견 1의 비율."""
    expr = [a.opinion for a in model.agents if a.expressing]
    return float(np.mean(expr)) if expr else float("nan")


def expressing_frac(model):
    return float(np.mean([a.expressing for a in model.agents]))


class Citizen(mesa.Agent):
    def __init__(self, model, opinion):
        super().__init__(model)
        self.opinion = opinion        # 사적 의견 0/1 (고정)
        self.expressing = True        # 표명 여부 (변함)

    def step(self):
        m = self.model
        others = [a for a in m.agents if a is not self and a.expressing]
        if not others:
            self.expressing = True
            return
        sample = m.random.sample(others, min(m.k, len(others)))
        same = sum(1 for a in sample if a.opinion == self.opinion)
        # =============================================================
        # TODO(학생): 지각·표명 규칙 (책 5장 §5.3)
        # 1) perceived = 표본(sample) 중 내 편(same) 비율
        # 2) perceived < m.threshold 이면 침묵 성향:
        #    self.expressing = m.random.random() > m.silence_prob
        #    아니면 self.expressing = True
        raise NotImplementedError("TODO: 규칙을 채우세요")
        # =============================================================


class SpiralOfSilence(mesa.Model):
    def __init__(self, n=120, p1=0.55, k=8, threshold=0.5, silence_prob=0.7, seed=None):
        super().__init__(seed=seed)
        self.k, self.threshold, self.silence_prob = k, threshold, silence_prob
        self.p1 = p1
        for _ in range(n):
            Citizen(self, 1 if self.random.random() < p1 else 0)
        self.datacollector = mesa.DataCollector(model_reporters={
            "expressed_share_1": expressed_share_1,
            "expressing_frac": expressing_frac})

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    m = SpiralOfSilence(n=120, p1=0.55, seed=1)
    df = run(m, 30)
    print(f"[서명] 실제 의견 1 비율 = {m.p1}  (슬쩍 다수)")
    print(f"  표명된 의견 1 비율: 초기 {df['expressed_share_1'].iloc[0]:.2f}"
          f" → 최종 {df['expressed_share_1'].iloc[-1]:.2f}  (나선으로 우세해짐)")
    print(f"  표명 비율: {df['expressing_frac'].iloc[0]:.2f} → {df['expressing_frac'].iloc[-1]:.2f} (침묵 증가)")
    plot_series(df, "w05_spiral.png", "spiral of silence (p1=0.55)")
    print("  그림 저장: w05_spiral.png")
