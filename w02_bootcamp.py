"""2주 · Python + AI 보조 시뮬레이션 부트캠프.

목표: 통계 R에서 시뮬레이션 Python으로 넘어오는 다리. ABM에 필요한 최소 기술만.
  반복문 · 함수 · 난수(seed) · 상태 갱신 루프 · 디버깅 · 그림.
그리고 이 과목의 핵심 = AI 보조 + 검증 워크플로.

실행: python w02_bootcamp.py  (그림 하나를 bootcamp_walk.png로 저장)
"""
from __future__ import annotations
import numpy as np

# =====================================================================
# 0. AI 보조 워크플로 (프롬프트 → 검증 → 해석)
# ---------------------------------------------------------------------
# 이 과목에서 코드는 AI로 짜도 된다. 단 반드시:
#   (1) 프롬프트: 무엇을 시켰나  (예: "eps 이내 이웃 평균으로 이동하는 함수")
#   (2) 검증:     출력이 맞는지 어떻게 확인했나
#                 - 극단값 점검(eps=0 이면 아무도 안 움직임 등)
#                 - 손계산 대조(작은 예제)
#                 - 재실행(같은 seed면 같은 결과)
#   (3) 해석:     결과가 이론에 대해 무엇을 뜻하나(그리고 뜻하지 않나)
# 제출물에는 '검증 로그 1문단'과 핵심 규칙 줄의 '자기 언어 주석'이 들어간다.
# =====================================================================


# 1. 반복문 + 함수 -----------------------------------------------------
def mean_of(xs):
    """리스트 평균. (numpy 없이 감을 잡기 위한 손 구현)"""
    total = 0.0
    for x in xs:          # for 반복문
        total += x
    return total / len(xs)


# 2. 난수 + seed 재현 --------------------------------------------------
def draw(n, seed):
    rng = np.random.default_rng(seed)   # seed 고정 = 재현 가능
    return rng.random(n)                # 0~1 난수 n개


# 3. 상태 갱신 루프 = ABM의 핵심 골격 ----------------------------------
def random_walk(steps, seed):
    """한 상태 변수를 매 단계 난수로 갱신하는 최소 시뮬레이션.
    ABM은 이 '상태 → 규칙 → 갱신' 루프를 여러 행위자로 확장한 것일 뿐이다.
    """
    rng = np.random.default_rng(seed)
    x = 0.0
    history = [x]
    for _ in range(steps):
        x += rng.normal(0, 1)   # 규칙: 정규난수만큼 이동
        history.append(x)       # 상태 기록
    return history


# 4. 디버깅 연습 -------------------------------------------------------
def buggy_share_positive(xs):
    """양수 비율을 반환해야 하는데 버그가 있다. 어디가 틀렸나?
    검증법: share_positive([1,-1]) 은 0.5 여야 한다.
    """
    count = 0
    for x in xs:
        if x > 0:
            count += 1
    return count            # 버그: len(xs)로 나눠야 '비율'


def share_positive(xs):     # 고친 버전
    return sum(1 for x in xs if x > 0) / len(xs)


if __name__ == "__main__":
    # 1
    print("mean_of([1,2,3,4]) =", mean_of([1, 2, 3, 4]))
    # 2 재현성 확인: 같은 seed면 같은 값
    assert np.allclose(draw(3, 7), draw(3, 7)), "seed 고정이면 같아야 한다"
    print("draw(3, seed=7) =", np.round(draw(3, 7), 3))
    # 3 상태 갱신 루프
    hist = random_walk(steps=200, seed=1)
    print(f"random_walk 200스텝: 시작 {hist[0]:.2f} → 끝 {hist[-1]:.2f}")
    # 4 디버깅: 검증으로 버그를 잡는다
    print("buggy:", buggy_share_positive([1, -1]), " (0.5 여야 하는데 틀림)")
    print("fixed:", share_positive([1, -1]))
    # 5 그림 저장 (matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(hist)
        plt.xlabel("step"); plt.ylabel("state x"); plt.title("random walk (seed=1)")
        plt.tight_layout(); plt.savefig("bootcamp_walk.png", dpi=110); plt.close()
        print("그림 저장: bootcamp_walk.png")
    except ImportError:
        print("(matplotlib 미설치 — requirements.txt 설치 후 그림)")

# =====================================================================
# 6. R → Python 빠른 대응 (이전기 R 코호트용 ; AI에게 번역도 시켜보라)
# ---------------------------------------------------------------------
#   R                              Python
#   ---------------------------    ------------------------------------
#   c(1,2,3)                       [1,2,3]  또는  np.array([1,2,3])
#   for (i in 1:n) {...}           for i in range(n): ...
#   function(x) x^2                def f(x): return x**2   /  lambda x: x**2
#   mean(x); sd(x)                 x.mean(); x.std()
#   set.seed(1); runif(3)          rng = np.random.default_rng(1); rng.random(3)
#   df$col ; dplyr::filter         df["col"] ; df[df.col > 0]   (pandas)
#   lm(y ~ x, data=d)              statsmodels: smf.ols("y ~ x", d).fit()
#   ggplot(...) + geom_line()      plt.plot(...)   또는  df.plot()
# =====================================================================
