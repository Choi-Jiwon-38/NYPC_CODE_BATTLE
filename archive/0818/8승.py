from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations
from collections import Counter

import sys

# ----------------------------
# 주의사항:
# 디버깅용 print 함수 코드가 들어가 있으므로
# 제출시에는 Ctrl+F로 "디버깅"을 찾은 뒤에
# 주석처리가 되어 있는지 확인 후 제출할 것!
# ----------------------------

DEBUG_MODE = False

# 가능한 주사위 규칙들을 나타내는 enum
class DiceRule(Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    CHOICE = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    YACHT = 11

# 각 규칙의 평균 기대 점수 (참고용)
AVERAGE_SCORES = {
    DiceRule.ONE: 5000, DiceRule.TWO: 6000, DiceRule.THREE: 9000,
    DiceRule.FOUR: 20000, DiceRule.FIVE: 25000, DiceRule.SIX: 24190,
    DiceRule.CHOICE: 27000, DiceRule.FOUR_OF_A_KIND: 19000, DiceRule.FULL_HOUSE: 22000,
    DiceRule.SMALL_STRAIGHT: 15000, DiceRule.LARGE_STRAIGHT: 30000, DiceRule.YACHT: 50000,
}

# 점수를 버려야 할 때의 희생 우선순위
SACRIFICE_PRIORITY = [
    # 가장 먼저 버릴 것들 (낮은 점수 + 높은 확률)
    DiceRule.ONE,      # 1은 점수가 낮고 버퍼로 사용
    DiceRule.TWO,      # 2도 낮은 점수
    DiceRule.THREE,    # 3도 낮은 점수
    DiceRule.YACHT,    # 필요하다면 버려야함

    # 중간 우선순위
    DiceRule.FOUR_OF_A_KIND,
    DiceRule.FULL_HOUSE,
    DiceRule.FOUR,     # 기본 규칙 4, 5, 6은 보너스를 위해 우선순위를 높게
    DiceRule.FIVE,
    DiceRule.SIX,

    # 낮은 우선순위
    DiceRule.CHOICE,
    DiceRule.SMALL_STRAIGHT,
    DiceRule.LARGE_STRAIGHT,
]

# ==================================================================== #
# 가중치 상수들
# 가중치 조정할 때 숫자를 변경하여 조절할 것!

W_YACHT = 4.0
W_LARGE_STRAIGHT = 1.5
W_SMALL_STRAIGHT = 1.2
W_DEMOTION = 0.8
W_HIGH_PROMOTION = 1.2
W_LOW_PROMOTION = 1.1
W_CHOICE = 0.01

# 숫자의 중요도
# NOTE: 일단 합연산으로 구현은 했으나, 이렇게 구현해도 괜찮을지 고민 필요
W_UP_IMPORTANT = 0.1
W_DOWN_IMPORTANT = -0.1
W_VERY_IMPORTANT = W_UP_IMPORTANT * 3
W_NOT_IMPORTANT = W_DOWN_IMPORTANT * 3
W_NUMBERS_INIT = [
    1.0,   # 1
    1.1,   # 2
    1.3,   # 3
    1.5,   # 4
    1.8,   # 5
    2.0    # 6
]  # 높은 숫자일수록 기본적으로 중요도가 높음

LOW_UTILITY = 0.01        # 효율성이 이 이하이면 SACRIFICE
SCORE_HIGH_FULLHOUSE = 22 # Full House가 좋은 선택일지 정하는 임계 점수
SCORE_HIGH_FOK = 19       # Four of Kind가 좋은 선택일지 정하는 임계 점수
NR_END_GAME = 2           # 게임 후반부인지 판단할 남은 Rule 개수의 임계값

# ====== 상대 성향 기반 하향 조정 파라미터 (다운스케일만) ======
OPP_RECENT_N = 4            # 최근 N개 라운드만 관찰
OPP_AVG_MULT = 1.25          # 평균 대비 허용 배수(여유분)
OPP_MARGIN_MIN = 50         # 최소 마진
OPP_MARGIN_MAX = 1500       # 최대 마진
OPP_MARGIN_FACTOR = 0.02    # margin = score_diff * 이 값 (clamp 적용)

# ====== 보너스 달성 후 고눈(5,6) 선호 강도 ======
HIGH_SUM_ALPHA = 0.6        # 합이 높을수록 utility를 키우는 강도(0~0.6 권장)
LOW_FACE_PENALTY = 0.85     # 보너스 후 ONE/TWO/THREE의 경향 완화 배수

# ==================================================================== #

# 입찰 방법을 나타내는 데이터클래스
@dataclass
class Bid:
    group: str  # 입찰 그룹 ('A' 또는 'B')
    amount: int  # 입찰 금액


# 주사위 배치 방법을 나타내는 데이터클래스
@dataclass
class DicePut:
    rule: DiceRule  # 배치 규칙
    dice: List[int]  # 배치할 주사위 목록


# 게임 상태를 관리하는 클래스
class Game:
    def __init__(self):
        self.my_state = GameState()  # 내 팀의 현재 상태
        self.opp_state = GameState() # 상대 팀의 현재 상태
        self.round = 0
        # 상대방 베팅 히스토리 (최근 10개만 유지)
        self.opp_bid_history: List[int] = []
        # 라운드 별 숫자의 중요도
        self.W_NUMBERS_ROUND: List[float] = []

    # ================================ [필수 구현] ================================

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        num_to_pick = 5
        group_a, group_b = dice_a + self.my_state.dice, dice_b + self.my_state.dice
        remaining_rules = sum(1 for s in self.my_state.rule_score if s is None)
        unique_combination_a = {tuple(sorted(comb)) for comb in combinations(group_a, num_to_pick)}
        unique_combination_b = {tuple(sorted(comb)) for comb in combinations(group_b, num_to_pick)}

        # 12 라운드(남은 규칙 2개 이하)인 경우: 점수의 최대합으로 계산
        final_group_a, final_group_b = [], []
        rule_a: Optional[DiceRule] = None
        rule_b: Optional[DiceRule] = None
        weight_a, weight_b = -1.0, -1.0

        if remaining_rules <= 2:
            for dice_combination in unique_combination_a:
                tmp_rule_a, tmp_weight_a = self.calculate_end_game(list(dice_combination), self.my_state)
                if tmp_weight_a > weight_a:
                    final_group_a = list(dice_combination)
                    rule_a, weight_a = tmp_rule_a, tmp_weight_a
            for dice_combination in unique_combination_b:
                tmp_rule_b, tmp_weight_b = self.calculate_end_game(list(dice_combination), self.my_state)
                if tmp_weight_b > weight_b:
                    final_group_b = list(dice_combination)
                    rule_b, weight_b = tmp_rule_b, tmp_weight_b
        else:
            for dice_combination in unique_combination_a:
                tmp_rule_a, tmp_weight_a = self.calculate_best_put(list(dice_combination), self.my_state)
                if tmp_weight_a > weight_a:
                    final_group_a = list(dice_combination)
                    rule_a, weight_a = tmp_rule_a, tmp_weight_a
            for dice_combination in unique_combination_b:
                tmp_rule_b, tmp_weight_b = self.calculate_best_put(list(dice_combination), self.my_state)
                if tmp_weight_b > weight_b:
                    final_group_b = list(dice_combination)
                    rule_b, weight_b = tmp_rule_b, tmp_weight_b

        # 각 그룹의 기대 점수 차이 계산
        score_a = 0 if rule_a is None else GameState.calculate_score(DicePut(rule_a, final_group_a))
        score_b = 0 if rule_b is None else GameState.calculate_score(DicePut(rule_b, final_group_b))
        score_diff = int(abs(score_a - score_b))

        amount: float = score_diff * 0.25

        low_basic_rule_list = [DiceRule.ONE, DiceRule.TWO, DiceRule.THREE]

        if rule_a in low_basic_rule_list and rule_b in low_basic_rule_list:
            amount *= 0.3

        if score_diff <= 2000:
            importance_a = sum(self.get_importance_of_numbers(group_a, self.my_state))
            importance_b = sum(self.get_importance_of_numbers(group_b, self.my_state))
            group = "A" if importance_a >= importance_b else "B"
            # 점수 기대가 비슷하면 베팅 금액을 대폭 감소
            amount *= 0.2
        else:
            # 일반적으로 더 좋은 쪽을 선택
            group = "A" if weight_a > weight_b else "B"

        # 동점이면(즉시 기대 동일) 0입찰 회피 + 미래가치 반영 앵커
        if score_diff == 0:
            if group == "A":
                for d in final_group_a:
                    if d in group_a:
                        group_a.remove(d)
                amount = sum(group_a) * 100
            else:
                for d in final_group_b:
                    if d in group_b:
                        group_b.remove(d)
                amount = sum(group_b) * 100

        # 1라운드에 상대 히스토리 없으면 작은 앵커 금액 설정(0 회피)
        if len(self.opp_bid_history) == 0:
            amount = sum(group_a) * 100 if group == "A" else sum(group_b) * 100

        # 상대 최근 베팅 성향(평균/최대) 기반 '하향 조정만' 적용
        amount = self._cap_bid_with_opp_stats(amount, score_diff)
        amount = max(0, min(100000, int(amount)))
        return Bid(group, amount)

    # 최근 상대 베팅을 바탕으로 amount가 과도하면 낮추는(캡) 함수
    def _cap_bid_with_opp_stats(self, amount: float, score_diff: int) -> int:
        # 최근 데이터가 충분치 않으면 그대로 반환(단, 0입찰 방지용 100 바닥)
        recent = self.opp_bid_history[-OPP_RECENT_N:]
        if len(recent) < 3:
            return int(max(100, amount))

        opp_avg = sum(recent) / len(recent)
        opp_max = max(recent)

        # score_diff 크기에 비례한 안전 마진 (과도하지 않도록 clamp)
        margin = max(OPP_MARGIN_MIN, min(OPP_MARGIN_MAX, int(score_diff * OPP_MARGIN_FACTOR)))

        # 평균과 최대치를 종합한 상한선: "평균의 여유배수"와 "최대치 + 마진" 중 더 큰 값
        cap_by_avg = opp_avg * OPP_AVG_MULT
        cap_by_max = opp_max + margin
        soft_cap = max(cap_by_avg, cap_by_max)

        new_amount = int(min(amount, soft_cap))
        return max(0, min(100000, new_amount))

    def calculate_put(self) -> DicePut:
        best_put: List[DicePut] = []
        max_weight = -1.0

        dice_pool = sorted(self.my_state.dice)
        num_to_pick = 5 if len(dice_pool) >= 5 else len(dice_pool)
        if num_to_pick == 0:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            return DicePut(rule_to_sacrifice, [])

        if DEBUG_MODE:
            print(f"{self.round}R, CALC START -> dice pool: {sorted(dice_pool)}", file=sys.stderr)

        # 규칙이 2개 남은 경우: 가중치 대신 실제 점수 최대
        remaining_rules = sum(1 for s in self.my_state.rule_score if s is None)
        unique_combination = {tuple(sorted(comb)) for comb in combinations(dice_pool, num_to_pick)}
        if remaining_rules <= 2:
            max_score = -1
            best_rule = None
            best_dice = []
            for dice_combination in unique_combination:
                current_rule, current_score = self.calculate_end_game(list(dice_combination), self.my_state)
                if current_score > max_score:
                    best_dice = list(dice_combination)
                    best_rule, max_score = current_rule, current_score
            if best_rule is None:
                # 안전장치: 조합이 비어있거나 계산 실패 시 희생 규칙으로 처리
                rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
                dice_to_sacrifice = self.get_victim_dices(dice_pool, num_to_pick, rule_to_sacrifice)
                return DicePut(rule_to_sacrifice, dice_to_sacrifice)
            return DicePut(best_rule, best_dice)
        else:
            # 모든 조합에 대해 max_weight인 조합을 선별
            for dice_combination in unique_combination:
                dice_list = list(dice_combination)
                best_rule, current_weight = self.calculate_best_put(dice_list, self.my_state)
                if current_weight > max_weight:
                    max_weight = current_weight
                    best_put = [DicePut(best_rule, dice_list)]
                elif len(best_put) > 0 and current_weight == max_weight:
                    best_put.append(DicePut(best_rule, dice_list))

            if DEBUG_MODE:
                print(f"{self.round}R, CALC END -> max weight: {max_weight}", file=sys.stderr)
                print(f"{self.round}R - best_put: {best_put}", file=sys.stderr)

            if len(best_put) == 0 or max_weight <= LOW_UTILITY:
                rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
                # 희생 다이스 계산
                dice_to_sacrifice = self.get_victim_dices(dice_pool, num_to_pick, rule_to_sacrifice)
                return DicePut(rule_to_sacrifice, dice_to_sacrifice)
            elif len(best_put) == 1:
                return best_put[0]
            else:
                # 여러 후보면 중요도 합이 가장 낮은 조합 선택(아껴야 할 눈을 남기기 위해)
                W_NUMBERS = self.get_importance_of_numbers(dice_pool, self.my_state)
                def importance_sum(dice_list: List[int]) -> float:
                    return sum(W_NUMBERS[val - 1] for val in dice_list)
                best_put.sort(key=lambda put: (importance_sum(put.dice), sum(put.dice)))
                return best_put[0]

    # ============================== [필수 구현 끝] ==============================

    def initialize_importance(self):
        # W_NUMBERS_ROUND는 라운드가 넘어갈 때마다 초기화
        self.W_NUMBERS_ROUND = list(W_NUMBERS_INIT)

    # 라운드 별 W_NUMBERS 리스트의 값을 변경하는 함수 (보류용)
    def update_importance_of_numbers(self, modified: List[int]) -> None:
        assert (modified is not None) and (len(modified) == 6)
        self.W_NUMBERS_ROUND = list(modified)

    # W_NUMBERS를 계산하는 함수 (합연산)
    def get_importance_of_numbers(self, dice: List[int], state: 'GameState') -> List[float]:
        # 숫자의 중요도(라운드 초기값 복사)
        W_NUMBERS = list(self.W_NUMBERS_ROUND)

        # 기본 규칙 중 사용 여부에 따라 가중 조절(현재 소유 주사위 분포 반영)
        _dice_count = [self.my_state.dice.count(i) for i in range(1, 7)]
        for number in range(1, 7):
            num_idx = number - 1
            if state.rule_score[num_idx] is None:
                W_NUMBERS[num_idx] += (W_UP_IMPORTANT * _dice_count[num_idx])
            else:
                W_NUMBERS[num_idx] += (W_DOWN_IMPORTANT * _dice_count[num_idx])

        # ----- 상황 인식 플래그 -----
        basic = sum(s for s in state.rule_score[0:6] if s is not None)
        upper_bonus = basic >= 63000
        unused_basic = sum(1 for i in range(6) if state.rule_score[i] is None)
        straights_done = (
            state.rule_score[DiceRule.LARGE_STRAIGHT.value] is not None and
            state.rule_score[DiceRule.SMALL_STRAIGHT.value] is not None
        )
        yacht_done = state.rule_score[DiceRule.YACHT.value] is not None
        combo_available = any(
            state.rule_score[r.value] is None for r in (
                DiceRule.CHOICE, DiceRule.FOUR_OF_A_KIND, DiceRule.FULL_HOUSE
            )
        )

        # 상단 보너스를 이미 받았거나 기본 규칙이 거의 끝났다면 -> 고눈(5,6) 선호, 저눈(1~3) 디모션
        if upper_bonus or unused_basic <= 1:
            W_NUMBERS[4] += 0.5  # face 5
            W_NUMBERS[5] += 0.7  # face 6
            W_NUMBERS[0] -= 0.4  # face 1
            W_NUMBERS[1] -= 0.25 # face 2
            W_NUMBERS[2] -= 0.15 # face 3

        # Yacht를 이미 사용했다면, 저눈을 '야추 버퍼'로 들고 있을 이유가 줄어듦 -> 저눈 추가 디모션
        if yacht_done:
            for face in (1, 2, 3):
                W_NUMBERS[face - 1] -= 0.1 * _dice_count[face - 1]

        # 남은 조합 규칙(Choice/FH/FOK)이 있다면 합이 큰 편이 유리 -> 고눈 소폭 버프
        if combo_available:
            # 두 스트레이트를 이미 썼다면, 합 위주 족보 비중이 더 커짐 → 가산치 추가
            extra = 0.1 if straights_done else 0.0
            W_NUMBERS[4] += 0.2 + extra
            W_NUMBERS[5] += 0.3 + extra

        # ----- Yacht / Straight 가능성 체크로 중요도 보정 -----
        _remaining_dice = list(state.dice)
        for d in dice:
            if d in _remaining_dice:
                _remaining_dice.remove(d)

        if len(_remaining_dice) >= 5:
            # 스트레이트를 모두 사용했다면, 더 이상 스트레이트 포텐 증가는 하지 않음
            if not straights_done:
                # 1) Yacht (아직 사용 전)
                if state.rule_score[DiceRule.YACHT.value] is None:
                    counts = Counter(_remaining_dice)
                    common_number, nr_common_number = counts.most_common(1)[0]
                    if nr_common_number >= 4:
                        W_NUMBERS[common_number - 1] += W_VERY_IMPORTANT

                # 2) Straight (둘 중 하나라도 남아있으면 체크)
                if (
                    state.rule_score[DiceRule.LARGE_STRAIGHT.value] is None or
                    state.rule_score[DiceRule.SMALL_STRAIGHT.value] is None
                ):
                    unique_remaining = sorted(list(set(_remaining_dice)))
                    if len(unique_remaining) >= 3:
                        has_potential = any(
                            unique_remaining[i + 1] == unique_remaining[i] + 1 and
                            unique_remaining[i + 2] == unique_remaining[i] + 2
                            for i in range(len(unique_remaining) - 2)
                        )
                        if has_potential:
                            for continuos_number in unique_remaining:
                                W_NUMBERS[continuos_number - 1] += W_VERY_IMPORTANT

        # 음수로 내려가지 않도록 하한선 보정(과도한 디모션 방지)
        for i in range(6):
            W_NUMBERS[i] = max(0.05, W_NUMBERS[i])

        return W_NUMBERS

    # 규칙 희생이 발생할 때, 희생 규칙에 따른 숫자 중요도로 희생 주사위 선택
    def get_victim_dices(self, dice: List[int], num_to_pick: int, sacrifice: DiceRule) -> List[int]:
        assert self.my_state.rule_score[sacrifice.value] is None

        if len(dice) <= 5:  # 1라운드 등
            return sorted(dice)

        # 희생 규칙을 임시로 사용 처리하여 중요도 계산 가능하게 함
        self.my_state.rule_score[sacrifice.value] = 0

        victim_list: List[int] = []
        for _ in range(num_to_pick):
            remaining_dices = list(dice)
            for d in victim_list:
                if d in remaining_dices:
                    remaining_dices.remove(d)

            W_NUMBERS = self.get_importance_of_numbers(remaining_dices, self.my_state)

            min_importance, victim_dice = sys.maxsize, -1
            for number in remaining_dices:
                num_idx = number - 1
                if W_NUMBERS[num_idx] < min_importance:
                    min_importance = W_NUMBERS[num_idx]
                    victim_dice = number

            assert victim_dice != -1
            victim_list.append(victim_dice)

        # 원복
        self.my_state.rule_score[sacrifice.value] = None
        return sorted(victim_list)

    # utility 계산 (곱연산)
    def _get_utility_of_rules(self, score: int, rule: DiceRule, dice: List[int], state: 'GameState') -> float:
        # 기본 Utility
        utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0

        # --- 기본 숫자 규칙 (ONE ~ SIX) ---
        if rule.value <= 5:
            dice_number = rule.value + 1
            count_of_number = dice.count(dice_number)
            count_of_number_all_dice = state.dice.count(dice_number)

            if dice_number in [3, 4, 5, 6]:
                if count_of_number == 5:
                    utility *= W_YACHT
                elif count_of_number >= 4:
                    utility *= W_HIGH_PROMOTION
                else:
                    if dice_number in [5, 6]:  # 5,6은 좀더 빡빡하게
                        utility *= 0.2 if count_of_number <= 2 else 0.4
                    else:
                        utility *= W_DEMOTION
            else:
                # ONE, TWO는 보너스 전 Yacht 대기 자원
                if self.my_state.rule_score[DiceRule.YACHT.value] is None:
                    if count_of_number_all_dice >= 5:
                        utility *= 0.1
                    elif count_of_number_all_dice == 4:
                        utility *= 0.2
                    else:
                        utility *= W_HIGH_PROMOTION
                else:
                    utility *= W_DEMOTION

        # --- 조합 규칙 ---
        else:
            dice_sum = sum(dice)
            if rule == DiceRule.YACHT:
                utility *= W_YACHT
            elif rule == DiceRule.LARGE_STRAIGHT:
                utility *= W_LARGE_STRAIGHT
            elif rule == DiceRule.SMALL_STRAIGHT:
                utility *= W_SMALL_STRAIGHT
            elif rule == DiceRule.FOUR_OF_A_KIND:
                utility *= (W_HIGH_PROMOTION if dice_sum >= SCORE_HIGH_FOK else W_DEMOTION)
            elif rule == DiceRule.FULL_HOUSE:
                utility *= (W_HIGH_PROMOTION if dice_sum >= SCORE_HIGH_FULLHOUSE else W_DEMOTION)
            elif rule == DiceRule.CHOICE:
                utility *= W_CHOICE

        return utility

    # 10개의 다이스 중 5:5로 나누었을 때의 점수 합을 반환하는 함수
    def calculate_end_game(self, dice: List[int], state: 'GameState') -> Tuple[DiceRule, int]:
        all_rules = [r for r in list(DiceRule) if state.rule_score[r.value] is None]

        # 13라운드(규칙 1개 남음)
        if len(all_rules) == 1:
            r = all_rules[0]
            return r, state.calculate_score(DicePut(r, dice))
        else:
            assert len(all_rules) == 2

        rule_a, rule_b = all_rules[0], all_rules[1]
        remaining_dice = list(state.dice)
        for d in dice:
            if d in remaining_dice:
                remaining_dice.remove(d)

        score_a = state.calculate_score(DicePut(rule_a, dice)) + \
                  state.calculate_score(DicePut(rule_b, remaining_dice))
        score_b = state.calculate_score(DicePut(rule_b, dice)) + \
                  state.calculate_score(DicePut(rule_a, remaining_dice))

        return (rule_a, score_a) if score_a >= score_b else (rule_b, score_b)

    # 현재 상황과 가중치를 고려하여 제일 좋은 put을 반환하는 Wrapper 함수
    def calculate_best_put(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], float]:
        best_rule: Optional[DiceRule] = None
        max_weight = -1.0

        all_rules = list(DiceRule)
        current_numbers = self.get_importance_of_numbers(dice, state)
        if DEBUG_MODE:
            print(f"{self.round}R, dice: {sorted(dice)}, importance: {current_numbers}", file=sys.stderr)

        basic_score = sum(s for i, s in enumerate(state.rule_score) if s and i <= 5)

        # 모든 규칙 후보 평가
        for rule in all_rules:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))
                if score <= 0:
                    continue

                # Bonus를 획득하지 않은 경우 기본 규칙에 importance 사용
                if basic_score < 63000 and rule.value <= 5:
                    remaining_basic = sum(1 for i in range(6) if state.rule_score[i] is None)
                    if rule.value <= 3:
                        importance = 1.5 + 0.1 * remaining_basic
                    else:
                        importance = 2.5 + 0.1 * remaining_basic
                else:
                    importance = 1.0

                tmp_utility = self._get_utility_of_rules(score, rule, dice, state)

                # 기본 보너스 달성 후에는 고눈(합이 큰 조합)을 선호하도록 버프
                if basic_score >= 63000:
                    if rule in (DiceRule.CHOICE, DiceRule.FULL_HOUSE, DiceRule.FOUR_OF_A_KIND):
                        s = sum(dice)
                        h = (s - 5) / 25.0  # 5..30 -> 0..1 정규화
                        tmp_utility *= (1.0 + HIGH_SUM_ALPHA * max(0.0, min(1.0, h)))
                    if rule in (DiceRule.ONE, DiceRule.TWO, DiceRule.THREE):
                        tmp_utility *= LOW_FACE_PENALTY

                utility = tmp_utility * importance

                if DEBUG_MODE:
                    base_util = score / AVERAGE_SCORES.get(rule, 1)
                    print(f"{self.round}R, score:{score//1000}, rule:{rule.name}, basic:{basic_score}, base:{base_util:.2f}, tmp:{tmp_utility:.3f}, imp:{importance:.2f}, total:{utility:.3f}", file=sys.stderr)

                if utility > max_weight:
                    max_weight, best_rule = utility, rule

        if max_weight != -1.0 and DEBUG_MODE:
            print(f"{self.round}R, calculate_best_put() END -> rule: {best_rule.name}, local_max_weight: {max_weight}", file=sys.stderr)
        return best_rule, max_weight


    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str):
        self.round += 1

        # 기존 상태 업데이트
        if my_group == "A":
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)
        else:
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)

        # 베팅 결과 반영
        self.my_state.bid(my_bid.group == my_group, my_bid.amount)
        self.opp_state.bid(opp_bid.group == ('B' if my_group == 'A' else 'A'), opp_bid.amount)

        # 상대방 베팅 히스토리 업데이트 (최근 10개만 유지)
        self.opp_bid_history.append(opp_bid.amount)
        if len(self.opp_bid_history) > 10:
            self.opp_bid_history.pop(0)

    def update_put(self, put: DicePut):
        self.my_state.use_dice(put)

    def update_set(self, put: DicePut):
        self.opp_state.use_dice(put)


class GameState:
    def __init__(self):
        self.dice: List[int] = []
        self.rule_score: List[Optional[int]] = [None] * 12
        self.bid_score = 0

    def get_total_score(self) -> int:
        basic = sum(s for s in self.rule_score[0:6] if s is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(s for s in self.rule_score[6:12] if s is not None)
        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int):
        self.bid_score += -amount if is_successful else amount

    def add_dice(self, new_dice: List[int]):
        self.dice.extend(new_dice)

    def use_dice(self, put: DicePut):
        if put.rule is None:
            return
        assert self.rule_score[put.rule.value] is None
        for d in put.dice:
            if d in self.dice:
                self.dice.remove(d)
        self.rule_score[put.rule.value] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        rule, dice = put.rule, sorted(put.dice)
        if not dice:
            return 0
        counts = {i: dice.count(i) for i in range(1, 7)}

        if rule.value <= 5:
            return counts[rule.value + 1] * (rule.value + 1) * 1000
        if rule == DiceRule.CHOICE:
            return sum(dice) * 1000
        if rule == DiceRule.FOUR_OF_A_KIND:
            return sum(dice) * 1000 if any(c >= 4 for c in counts.values()) else 0
        if rule == DiceRule.FULL_HOUSE:
            vals = counts.values()
            return sum(dice) * 1000 if (3 in vals and 2 in vals) or 5 in vals else 0
        unique_dice_str = "".join(map(str, sorted(list(set(dice)))))
        if rule == DiceRule.SMALL_STRAIGHT:
            return 15000 if ("1234" in unique_dice_str or "2345" in unique_dice_str or "3456" in unique_dice_str) else 0
        if rule == DiceRule.LARGE_STRAIGHT:
            return 30000 if ("12345" in unique_dice_str or "23456" in unique_dice_str) else 0
        if rule == DiceRule.YACHT:
            return 50000 if 5 in counts.values() else 0
        return 0


def main():
    game = Game()

    # 입찰 라운드에서 나온 주사위들
    dice_a, dice_b = [0] * 5, [0] * 5
    # 내가 마지막으로 한 입찰 정보
    my_bid = Bid("", 0)

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                # 게임 시작
                print("OK")
                continue

            if command == "ROLL":
                # 주사위 굴리기 결과 받기
                str_a, str_b = args
                for i, c in enumerate(str_a):
                    dice_a[i] = int(c)
                for i, c in enumerate(str_b):
                    dice_b[i] = int(c)

                # 경매 시작 -> 라운드 초기화
                game.initialize_importance()

                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
                continue

            if command == "GET":
                # 주사위 받기
                get_group, opp_group, opp_score = args
                opp_score = int(opp_score)
                game.update_get(
                    dice_a, dice_b, my_bid, Bid(opp_group, opp_score), get_group
                )
                continue

            if command == "SCORE":
                # 주사위 골라서 배치하기
                put = game.calculate_put()
                game.update_put(put)
                assert put.rule is not None
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                continue

            if command == "SET":
                # 상대의 주사위 배치
                rule, str_dice = args
                dice = [int(c) for c in str_dice]
                game.update_set(DicePut(DiceRule[rule], dice))
                continue

            if command == "FINISH":
                # 게임 종료
                break

            # 알 수 없는 명령어 처리
            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()
