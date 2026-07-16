"""트랙 C · 네트워크 확산·복합전염 — 기말 프로젝트 시작 코드 (self-contained, Mesa 3.x + networkx).

요지: 단순 전염은 이웃 1명이면 채택. 복합 전염은 k명 이상 필요(사회적 확증).
ODD-lite: 목적=복합 전염은 네트워크 구조에 어떻게 반응하나 / 행위자=노드 /
  상태=미채택/채택 / 환경=networkx 위상 / 규칙=채택 이웃>=threshold면 채택 /
  시간=동시 갱신 / 출력=최종 채택 비율 / 한계=고정 임계·잡음 없음.
서명: 복합(threshold>=2)은 무작위 지름길(rewire)이 늘수록 확산이 무너진다. 단순은 무관.

실행: python track_c_network_contagion.py
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import mesa


class Node(mesa.Agent):
    def __init__(self, model, node_id, seeded):
        super().__init__(model)
        self.node_id = node_id
        self.adopted = seeded
        self._next = seeded

    def step(self):
        if self.adopted:
            self._next = True
            return
        m = self.model
        n_ad = sum(m.node_agent[j].adopted for j in m.G.neighbors(self.node_id))
        if n_ad >= m.threshold:
            self._next = True


class NetworkContagion(mesa.Model):
    def __init__(self, n=400, k_deg=8, rewire=0.0, threshold=1, n_seed=12, seed=None):
        super().__init__(seed=seed)
        self.threshold = threshold
        self.G = nx.watts_strogatz_graph(n, k_deg, rewire, seed=seed)
        self.node_agent = {node: Node(self, node, node < n_seed) for node in self.G.nodes()}

    def step(self):
        for a in self.agents:
            a.step()
        for a in self.agents:
            a.adopted = a._next


def adopted_frac(model):
    return float(np.mean([a.adopted for a in model.agents]))


if __name__ == "__main__":
    print("전염유형   rewire=0.0  rewire=0.5")
    for name, thr in (("단순(1)", 1), ("복합(3)", 3)):
        row = []
        for rw in (0.0, 0.5):
            m = NetworkContagion(threshold=thr, rewire=rw, seed=1)
            for _ in range(60):
                m.step()
            row.append(adopted_frac(m))
        print(f"  {name}   {row[0]:8.2f}  {row[1]:8.2f}")

# 확장 아이디어(TODO): 척도 없는 망(barabasi_albert) / 이질적 임계(사람마다 다름) /
#   seeding 전략(허브 vs 무작위 vs 인접) / 확률적 전염(p<1) / 망각(채택 취소).
