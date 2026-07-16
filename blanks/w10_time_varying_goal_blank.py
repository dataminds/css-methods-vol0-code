# ===== 빈칸판(학생 배포용) =====
# TODO 블록을 채우고 raise 줄을 지우면, 완성본과 같은 서명이 나와야 합니다.
# 검증: 극단값 시험 + 씨앗 73 재실행 + (가능하면) 손 검산. 검증 로그 한 문단!

"""10주 · 인지 행위자 맛보기: 시변 목표 (기분관리 vs 기분조정) (Mesa 3.x).

CAM 브라우저 랩과 같은 개념의 최소판 (선택적 노출의 published 계보).
상태 = 기분 x(정서가, -1..1). 목표 T(t):
    기분관리(M0): 고정 유쾌 목표 (지금 기분만 최적화)
    기분조정(M1): 유쾌 → 과제 목표로 시간에 따라 이동 (앞으로 닥칠 과제에 대비)
    λ(t) = (t / t*)^γ ,  T(t) = (1-λ)·T_유쾌 + λ·T_과제   (M0은 λ=0 고정)
선택 = '예측 다음 기분'이 현재 목표 T(t)에 가장 가까운 항목을 softmax로.
서명: M1은 초반 유쾌(기분 회복) → 후반 과제 방향으로 선택이 이동 ; M0은 유쾌에 머묾.

⭐ 능동추론·AFT 같은 개발 중 이론은 다루지 않는다(대학원 소관). 여기선 최소 규칙만.
실행: python w10_time_varying_goal.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import mesa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 상위 폴더(materials) abmlib 임포트
from abmlib import run, plot_series


def softmax(values, tau):
    v = np.asarray(values, dtype=float) / tau
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


class Chooser(mesa.Agent):
    def __init__(self, model, mood):
        super().__init__(model)
        self.mood = mood            # 상태: 기분 정서가
        self.chosen_valence = 0.0

    def step(self):
        m = self.model
        T = m.target()                                  # 현재 목표 (시변)
        pred = [self.mood + m.alpha * (v - self.mood) for v in m.items]   # 예측 다음 기분
        value = [-abs(p - T) for p in pred]             # 목표에 가까울수록 좋음
        idx = m.random.choices(range(len(m.items)), weights=softmax(value, m.tau))[0]
        self.chosen_valence = m.items[idx]
        # =============================================================
        # TODO(학생): 기분 갱신 규칙 (책 10장 §10.2)
        # 힌트: 고른 항목(m.items[idx]) 쪽으로, 보폭 m.alpha 만큼
        # 꼴: self.mood += m.alpha * ( ??? - self.mood )
        raise NotImplementedError("TODO: 규칙을 채우세요")
        # =============================================================


class MoodModel(mesa.Model):
    def __init__(self, n=150, adjust=True, alpha=0.5, tau=0.15,
                 t_star=14, gamma=2.0, seed=None):
        super().__init__(seed=seed)
        self.adjust = adjust
        self.alpha, self.tau = alpha, tau
        self.t_star, self.gamma = t_star, gamma
        self.T_hed, self.T_task = 0.6, -0.6
        self.items = list(np.linspace(-1, 1, 9))
        self.t = 0
        for _ in range(n):
            Chooser(self, mood=-0.6)                    # 초기: 나쁜 기분 → 회복 동기
        self.datacollector = mesa.DataCollector(model_reporters={
            "chosen_valence": lambda mm: float(np.mean([a.chosen_valence for a in mm.agents]))})

    def target(self):
        if not self.adjust:
            return self.T_hed                            # M0: 유쾌 목표 고정
        lam = min(1.0, (self.t / self.t_star) ** self.gamma)
        return (1 - lam) * self.T_hed + lam * self.T_task

    def step(self):
        self.agents.shuffle_do("step")
        self.t += 1
        self.datacollector.collect(self)


if __name__ == "__main__":
    print("[서명] 기분조정(M1)은 초반 유쾌→후반 과제 방향 ; 기분관리(M0)는 유쾌에 머묾")
    for adjust, name in ((False, "M0 기분관리"), (True, "M1 기분조정")):
        df = run(MoodModel(adjust=adjust, seed=1), 14)
        early = df["chosen_valence"].iloc[:3].mean()
        late = df["chosen_valence"].iloc[-3:].mean()
        print(f"  {name}: 선택 정서가 초반 {early:+.2f} → 후반 {late:+.2f}")
    plot_series(run(MoodModel(adjust=True, seed=1), 14),
                "w10_mood.png", "mood adjustment: chosen valence over time")
    print("  그림 저장: w10_mood.png")
