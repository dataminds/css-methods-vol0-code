"""트랙 D · 캠페인 메시지 반응(부메랑) — 기말 프로젝트 시작 코드 (self-contained, Mesa 3.x).

요지: 캠페인은 반복 노출로 태도를 밀지만, 과다 노출은 피로·반발(부메랑)을 낳는다.
ODD-lite: 목적=노출 강도와 태도 변화의 관계 / 행위자=수용자 / 상태=태도+피로 /
  자극=캠페인 메시지(목표 방향) / 규칙: 노출마다 태도를 목표 쪽으로 (1-2*피로) 배로 이동
  (피로<0.5 설득, >0.5 반발) ; 피로는 노출로 쌓이고 조금 회복 / 출력=평균 태도 /
  한계=단일 캠페인·동질 반발.
서명: 노출 강도-반응이 뒤집힌 U자 = 적정 노출은 설득, 과다 노출은 부메랑(음의 태도).

실행: python track_d_campaign_response.py
"""
from __future__ import annotations
import numpy as np
import mesa


class Receiver(mesa.Agent):
    def __init__(self, model, attitude):
        super().__init__(model)
        self.att = attitude
        self.fatigue = 0.0

    def step(self):
        m = self.model
        for _ in range(m.exposures):                     # 스텝당 노출 횟수 = 강도
            eff = m.k * (1 - 2 * self.fatigue)            # 피로 높으면 반발(부호 역전)
            self.att = float(np.clip(self.att + eff * (m.target - self.att), -1, 1))
            self.fatigue = min(1.0, self.fatigue + m.df)
        self.fatigue = max(0.0, self.fatigue - m.decay)   # 조금 회복


class Campaign(mesa.Model):
    def __init__(self, n=200, exposures=2, k=0.10, target=1.0, df=0.02, decay=0.02, seed=None):
        super().__init__(seed=seed)
        self.exposures, self.k, self.target = exposures, k, target
        self.df, self.decay = df, decay
        for _ in range(n):
            Receiver(self, self.random.uniform(-0.2, 0.2))

    def step(self):
        self.agents.shuffle_do("step")


def mean_att(model):
    return float(np.mean([a.att for a in model.agents]))


if __name__ == "__main__":
    print("[서명] 노출 강도 vs 최종 태도 (목표 +1) : 적정↑, 과다→부메랑")
    for exp in (1, 2, 4, 6, 10):
        m = Campaign(exposures=exp, seed=1)
        for _ in range(8):
            m.step()
        print(f"  노출/스텝 {exp:2d}: 최종 평균태도 {mean_att(m):+.2f}")

# 확장 아이디어(TODO): 이질적 반발 성향 / 표적화(태도별 다른 메시지) / 경쟁 캠페인 2개 /
#   메시지 강도 vs 빈도 분리 / 회복(decay) 조작 = 캠페인 간격 효과.
