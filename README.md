# 계산으로 보는 사회 — 실습 코드

책 「계산으로 보는 사회: 파이썬 행위자 모형 첫걸음」(계산사회과학 방법 총서 0권)의 공식 실습 코드 저장소입니다.

- 책(무료 웹판, ALPHA): https://grow.minds.kr/textbooks/css-methods/vol0/
- 지은이: 안도현 (Ahn, Dohyun)

## 설치 (10분)

1. 파이썬 3.11 이상 설치 (python.org ; "Add Python to PATH" 체크)
2. 이 저장소 내려받기: 우측 상단 Code → Download ZIP (또는 `git clone`)
3. 터미널에서:

```bash
pip install -r requirements.txt
python -c "import mesa; print(mesa.__version__)"   # 3.3 이상이면 준비 끝
```

막히면 오류 메시지를 통째로 생성 AI에게 붙여넣고 물어보세요. 표준 작업 절차입니다(책 1장).

## 구성

| 폴더/파일 | 내용 | 책 연결 |
|---|---|---|
| `w02_bootcamp.py` ~ `w11_recovery_sensitivity.py` | 장별 모형 완성본 (실행·서명 검증됨) | 1~11장 |
| `blanks/` | 빈칸판: 심장 규칙을 TODO로 비운 연습판. 채우기 전엔 일부러 오류를 냅니다 | 각 장 실습 |
| `worksheets/` | 손계산·명세서·검증 로그·발표 양식 | 7·12·15장, 부록 B |
| `project-starters/` | 기말 프로젝트 트랙 시작 코드 5종 (자족적 ; 복사해 확장) | 14장 |

## 실행

```bash
python w03_bounded_confidence.py     # 각 파일 하단에 설계 점검·서명 출력
```

빈칸판으로 연습하려면 `blanks/`의 파일을 열어 TODO를 채우고, 완성본과 서명을 대조하세요.

## 재현성 규약

모든 확률적 결과는 씨앗(seed)을 명시합니다. 이 책의 기본 씨앗은 73, 교차 확인은 37입니다(서로 뒤집은 소수입니다). 결과를 보고할 때 씨앗 없이 보고하지 않기 — 책의 첫 규율입니다.

## 라이선스

- 코드: MIT (자유롭게 사용·수정·재배포)
- 책 본문(웹판): CC BY-NC 4.0 예정 (비상업 공유·개작 허용)

질문·오류 제보: help@minds.kr
