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
    DiceRule.ONE: 2880, DiceRule.TWO: 7280, DiceRule.THREE: 11570,
    DiceRule.FOUR: 16160, DiceRule.FIVE: 20690, DiceRule.SIX: 25190,
    DiceRule.CHOICE: 27000, DiceRule.FOUR_OF_A_KIND: 17100, DiceRule.FULL_HOUSE: 22590,
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
    DiceRule.FOUR,     # 기본 규칙 4, 5, 6의 경우 보너스 점수를 위해 우선순위를 높게 판단
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

W_YACHT = 2.0
W_LARGE_STRAIGHT = 1.5
W_SMALL_STRAIGHT = 1.3
W_DEMOTION = 0.8
W_HIGH_PROMOTION = 1.2
W_LOW_PROMOTION = 1.1
W_NICE_CHOICE = 1.15
W_BAD_CHOICE = 0.9

# 숫자의 중요도
# NOTE: 일단 합연산으로 구현은 했으나, 이렇게 구현해도 괜찮을지 고민 필요
W_UP_IMPORTANT = 0.2     
W_DOWN_IMPORTANT = -0.2  
W_VERY_IMPORTANT = W_UP_IMPORTANT * 3
W_NOT_IMPORTANT = W_DOWN_IMPORTANT * 3
W_NUMBERS_INIT = [
    1.0,   # 1
    1.1,   # 2
    1.3,   # 3
    1.5,   # 4
    1.8,   # 5
    2.0    # 6
] # 높은 숫자일수록 기본적으로 중요도가 높음

LOW_UTILITY = 0.01        # 효율성이 이 이하이면 SACRIFICE
BID_DIFF_RATE = 0.02      # 경매 입찰 시 입찰 점수 보정 비율
SCORE_GOOD_CHOICE = 27    # W_NICE_CHOICE를 적용할 임계 점수
SCORE_HIGH_FULLHOUSE = 22 # Full House가 좋은 선택일지 정하는 임계 점수
SCORE_HIGH_FOK = 18       # Four of Kind가 좋은 선택일지 정하는 임계 점수
NR_END_GAME = 2           # 게임 후반부인지 판단할 남은 Rule 개수의 임계값

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
        self.my_state = GameState() # 내 팀의 현재 상태
        self.opp_state = GameState() # 상대 팀의 현재 상태
        self.round = 0
        # 상대방 베팅 히스토리 (최근 10개만 유지)
        self.opp_bid_history = []
        # 라운드 별 숫자의 중요도
        self.W_NUMBERS_ROUND = []

    # ================================ [필수 구현] ================================

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        num_to_pick = 5
        group_a, group_b = dice_a + self.my_state.dice, dice_b + self.my_state.dice
        remaining_rules = sum(1 for s in self.my_state.rule_score if s is None)
        unique_combination_a = {tuple(sorted(comb)) for comb in combinations(group_a, num_to_pick)}
        unique_combination_b = {tuple(sorted(comb)) for comb in combinations(group_b, num_to_pick)}
        
        # 12 라운드인 경우 점수의 최대합으로 계산
        final_group_a, final_group_b = [], []
        tmp_rule_a, tmp_rule_b = None, None
        rule_a, rule_b = None, None
        if remaining_rules <= 2:
            weight_a, weight_b = -1, -1
            tmp_weight_a, tmp_weight_b = -1, -1
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
        # 각 주사위 묶음의 최대 기대 가중치를 계산
        else:
            weight_a, weight_b = -1.0, -1.0
            tmp_weight_a, tmp_weight_b = -1.0, -1.0
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
        score_diff = int(abs(score_a - score_b) * BID_DIFF_RATE)

        # 더 높은 효율을 가진 그룹에 입찰
        if rule_a is not None and weight_a > weight_b:
            group = "A"
            score = score_a
        elif rule_b is not None and weight_b > weight_a:
            group = "B"
            score = score_b
        else:
            # dice_a, dice_b의 우열이 없으면, 전체 다이스에서
            # 중요도가 높은 숫자들이 많은 그룹에 입찰
            importance_a = sum(self.get_importance_of_numbers(group_a, self.my_state))
            importance_b = sum(self.get_importance_of_numbers(group_b, self.my_state))
            group = "A" if importance_a >= importance_b else "B"
            score = 0

        # TODO: 승기를 잡고있다는 기준을 현재 점수가 아닌 기대 점수에 따른 방향으로 변경해야함.
        # 상대방 히스토리 기반 베팅 금액 계산 
        # + 기대 점수의 차이 값으로 보정
        if self.opp_bid_history:
            # 상대방 베팅 히스토리 분석
            max_opp_bid = 1 if max(self.opp_bid_history) == 0 else max(self.opp_bid_history)
            sorted_bids = sorted(self.opp_bid_history, reverse=True)
            top_3_avg = 1 if sum(sorted_bids[1:4]) / min(3, len(sorted_bids)) == 0 else sum(sorted_bids[1:4]) / min(3, len(sorted_bids))
            avg_opp_bid = 1 if sum(self.opp_bid_history) / len(self.opp_bid_history) == 0 else sum(self.opp_bid_history) / len(self.opp_bid_history)

            # 점수에 따른 베팅 전략
            if score >= 50000:  # Yacht - 가장 공격적
                amount = int(max_opp_bid * 1.2)  # 최대 베팅의 1.2배
            elif score >= 30000:  # Large Straight - 매우 공격적
                amount = int(top_3_avg * 1.1)  # 상위 3개 평균의 1.1배
            elif score >= 15000:  # Small Straight - 공격적
                amount = int(top_3_avg * 1.05)  # 상위 3개 평균의 1.05배
            elif score >= 10000:  # 높은 기본 점수 - 적당히 공격적
                amount = int(avg_opp_bid * 1.1)  # 평균 베팅의 1.1배
            else:
                amount = int(avg_opp_bid * 0.5)  # 평균 베팅의 0.5배
            
            # 보정 값 추가
            amount += score_diff
        else:
            amount = 0
        
        return Bid(group, amount)
    
    def calculate_put(self) -> DicePut:
        best_put = []
        max_weight = -1.0
        
        dice_pool = sorted(self.my_state.dice)
        num_to_pick = 5 if len(dice_pool) >= 5 else len(dice_pool)
        if num_to_pick == 0:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            return DicePut(rule_to_sacrifice, [])
        
        if DEBUG_MODE: print(f"{self.round}R, CALC START -> dice pool: {sorted(dice_pool)}", file=sys.stderr) # 디버깅용

        # 규칙이 2개 남은 경우 가중치를 계산하지 않고, 남은 주사위로 
        # 12, 13 라운드에서 가장 큰 점수를 얻을 수 있는 조합을 선별하여 반환.
        remaining_rules = sum(1 for s in self.my_state.rule_score if s is None)
        unique_combination = {tuple(sorted(comb)) for comb in combinations(dice_pool, num_to_pick)}
        if remaining_rules <= 2:
            max_score = -1
            for dice_combination in unique_combination:
                current_rule, current_score = self.calculate_end_game(list(dice_combination), self.my_state)
                if current_score > max_score:
                    best_put = list(dice_combination)
                    best_rule, max_score = current_rule, current_score
            
            return DicePut(best_rule, best_put)

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

            if DEBUG_MODE: print(f"{self.round}R, CALC END -> max weight: {max_weight}", file=sys.stderr); print(f"{self.round}R - best_put: {best_put}", file=sys.stderr) # 디버깅 용도
            # NOTE: 우리의 기저 전략이 Bonus의 달성 유무에 따라 라운드의 사용 가중치 종류가
            #       다름. i.e.) 라운드마다 utility만 사용 또는 utility * importance 사용이 발생함
            if len(best_put) == 0 or max_weight <= LOW_UTILITY:
                rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)

                # 희생 다이스 계산
                dice_to_sacrifice = self.get_victim_dices(dice_pool, num_to_pick, rule_to_sacrifice)
                return DicePut(rule_to_sacrifice, dice_to_sacrifice)

            # best_put이 단일 후보인 경우
            elif len(best_put) == 1:
                return best_put[0]
            
            # best_put의 여러 후보 중 중요도 합이 가장 낮은 것을 선택
            # NOTE: 생각해보니 여기도 dice_pool을 사용해서
            #       전체 dice 리스트에 대한 숫자의 중요도를 계산하면 되는거였음!
            else:
                W_NUMBERS = self.get_importance_of_numbers(dice_pool, self.my_state)
                def importance_sum(dice_list):
                    return sum(W_NUMBERS[val - 1] for val in dice_list)
                best_put.sort(key=lambda put: (importance_sum(put.dice), sum(put.dice)))
                return best_put[0]
    
    # ============================== [필수 구현 끝] ==============================

    def initialize_importance(self):
        # W_NUMBERS_ROUND는 라운드가 넘어갈 때마다 초기화
        self.W_NUMBERS_ROUND = list(W_NUMBERS_INIT)

    # 라운드 별 W_NUMBERS 리스트의 값을 변경하는 함수
    # NOTE: 현재 중요도 구조 상 사용되지 않으나,
    #       추후, 사용될 수 있기에 삭제하지 않음.
    def update_importance_of_numbers(self, modified: List[int]) -> None:
        assert (modified is not None) and (len(modified) == 6)

        self.W_NUMBERS_ROUND = list(modified)

    # W_NUMBERS를 계산하는 함수
    # W_NUMBER 값들은 합연산으로 계산
    def get_importance_of_numbers(self, dice: List[int], state: 'GameState') -> list[float]:
        # * 가중치 올림 -> 선택할 가능성 증가, 버릴 가능성 감소
        # * 가중치 내림 -> 선택할 가능성 감소, 버릴 가능성 증가

        # 숫자의 중요도
        W_NUMBERS = list(self.W_NUMBERS_ROUND)

        # 기본 규칙 중 사용하지 않은 숫자는 많은만큼 가중치 올림
        # 반대로 사용한 숫자는 가중치 내림
        _dice_count = [self.my_state.dice.count(i) for i in range(1, 7)]
        for number in range(1, 7):
            num_idx = number - 1
            if state.rule_score[num_idx] is None:
                W_NUMBERS[num_idx] += (W_UP_IMPORTANT * _dice_count[num_idx])
            else:
                W_NUMBERS[num_idx] += (W_DOWN_IMPORTANT * _dice_count[num_idx])

        # Yacht와 Straight 규칙 확인
        # 규칙이 실현된 가능성이 높은 경우 버리거나 사용하지 않도록 가중치 올림
        _remaining_dice = list(state.dice) 
        for d in dice:
            if d in _remaining_dice:
                _remaining_dice.remove(d)

        if len(_remaining_dice) >= 5: # 남아있는 주사위가 5개 이상일 때 (1라운드, 13라운드 제외)
            # 1. Yacht
            if state.rule_score[DiceRule.YACHT.value] is None:
                counts = Counter(_remaining_dice)
                common_number, nr_common_number = counts.most_common(1)[0][0], counts.most_common(1)[0][1]
            
                # 남은 주사위에 같은 숫자가 4개 이상이면, 다음 턴 Yacht를 기대
                if nr_common_number >= 4:
                    W_NUMBERS[common_number - 1] += W_VERY_IMPORTANT
            
            # 2. Straight
            if state.rule_score[DiceRule.LARGE_STRAIGHT.value] is None and \
                state.rule_score[DiceRule.SMALL_STRAIGHT.value] is None:

                # 남은 주사위가 스트레이트를 만들기 좋다면, 스트레이트 규칙을 아껴둠
                # 남은 주사위 중 다른 숫자가 4개 이상이고, 그 중 3개 이상이 연속되는지 확인
                # NOTE: L_ST와 S_ST의 가능성을 구분하여 계산할 수도 있음, i.e) 연속 3개와 4개를 각각 계산
                unique_remaining = sorted(list(set(_remaining_dice)))
                if len(unique_remaining) >= 3:
                    has_potential = False
                    # 3칸 연속 (예: 1,2,3 또는 2,3,4)이 있는지 확인
                    for i in range(len(unique_remaining) - 2):
                        if unique_remaining[i+1] == unique_remaining[i] + 1 and \
                            unique_remaining[i+2] == unique_remaining[i] + 2:
                            has_potential = True
                            break
                            
                    if has_potential:
                        for continuos_number in unique_remaining:
                            W_NUMBERS[continuos_number - 1] += W_VERY_IMPORTANT

        # NOTE: 중요도에 영향을 줄만한 상황이 있다면, 추가 필요

        return W_NUMBERS
    
    # 규칙 희생이 발생할 때, 희생 규칙에 따른
    # 숫자의 중요도를 계산하여 반환하는 Wrapper 함수
    def get_victim_dices(self, dice: List[int], num_to_pick: int, sacrifice: DiceRule) -> List[int]:
        assert self.my_state.rule_score[sacrifice.value] is None

        # 1라운드인 경우 남은 다이스를 반환
        if len(dice) <= 5:
            return sorted(dice)

        # 희생할 규칙을 사용한 규칙으로 보고 중요도를 계산할 수 있도록
        # 임시로 None이 아닌 값으로 변경
        self.my_state.rule_score[sacrifice.value] = 0

        victim_list = []
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
        
        # state의 rule_score를 변경하는 것은 이 함수의
        # 역할이 아니므로, 다시 None 값으로 설정
        self.my_state.rule_score[sacrifice.value] = None
        return sorted(victim_list)

    # utility를 계산하는 내부 함수
    # utility는 곱연산으로 계산
    def _get_utility_of_rules(self, score: int, rule: DiceRule, dice: List[int], state: 'GameState') -> float:
        # --- utility 계산 우선 순위 ---
        # 1. 야추
        # 2. FOUR, FIVE, SIX 적어도 3개이상
        # 3. 숫자가 큰 Full House, Four of kind
        # 4. Large Straight, Small Straight
        # 5. 숫자가 작은 Full House, Four of kind
        # 6. ONE, TWO ,THREE (상황이 마땅치 않을 때 0으로 버릴 가능성이 높음)
        # -------------------------------

        # NOTE: 일단 기존의 utility 계산 구조를 그대로 사용
        #       개선의 여지가 남아있을 가능성 높음!

        # 기본 Utility 계산
        utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0
                
        # 우선순위에 따른 전략적 가중치 적용
        # --- 기본 숫자 규칙 (ONE ~ SIX) ---
        if rule.value <= 5:
            dice_number = rule.value + 1
            count_of_number = dice.count(dice_number)

            # 우선순위 2번: FOUR, FIVE, SIX
            # 4개 이상이면 큰 가중치를, 5개 이상이면 매우 큰 가중치 적용
            if dice_number in [3, 4, 5, 6]:
                if count_of_number == 5:
                    utility *= W_YACHT
                elif count_of_number >= 4:
                    utility *= W_HIGH_PROMOTION
                else:
                    utility *= W_DEMOTION
                    
            # 그 외 나머지 기본 규칙(ONE, TWO, THREE)은 낮은 가중치 적용
            else:
                utility *= W_DEMOTION

        # --- 조합 규칙 ---
        else:
            dice_sum = sum(dice)
            # 우선순위 1번: Yacht
            if rule == DiceRule.YACHT:
                utility *= W_YACHT
            # 우선순위 4번: Large/Small Straight
            elif rule == DiceRule.LARGE_STRAIGHT:
                utility *= W_LARGE_STRAIGHT
            elif rule == DiceRule.SMALL_STRAIGHT:
                utility *= W_SMALL_STRAIGHT
            # 우선순위 3, 5번: Full House, Four of a Kind
            elif rule == DiceRule.FOUR_OF_A_KIND:
                if dice_sum >= SCORE_HIGH_FOK:
                    utility *= W_HIGH_PROMOTION
                else:
                    utility *= W_DEMOTION
            elif rule == DiceRule.FULL_HOUSE:
                if dice_sum >= SCORE_HIGH_FULLHOUSE:
                    utility *= W_HIGH_PROMOTION
                else:
                    utility *= W_DEMOTION
            # Choice 규칙
            # TODO: 초이스 규칙을 좀 더 고급 짬통으로 만들어야 함.
            elif rule == DiceRule.CHOICE:
                if dice_sum >= SCORE_GOOD_CHOICE:
                    utility *= W_NICE_CHOICE
                else:
                    utility *= W_BAD_CHOICE

        return utility

    # 10개의 다이스 중 5:5로 나누었을 때의 점수 합을 반환하는 함수
    def calculate_end_game(self, dice: List[int], state: 'GameState') -> Tuple[DiceRule, int]:
        all_rules = []
        for rule in list(DiceRule):
            if state.rule_score[rule.value] is None:
                all_rules.append(rule)
        
        # 13라운드인 경우 남은 다이스의 점수를 반환
        if len(all_rules) == 1:
            return all_rules[0], state.calculate_score(DicePut(all_rules[0], dice))
        else:
            assert len(all_rules) == 2

        rule_a, rule_b = all_rules[0], all_rules[1]
        remaining_dice = list(state.dice) 
        for d in dice:
            if d in remaining_dice:
                remaining_dice.remove(d)
        
        score_a = state.calculate_score(DicePut(rule_a, dice)) \
              + state.calculate_score(DicePut(rule_b, remaining_dice))
        score_b = state.calculate_score(DicePut(rule_b, dice)) \
              + state.calculate_score(DicePut(rule_a, remaining_dice))

        if score_a >= score_b:
            return rule_a, score_a
        else:
            return rule_b, score_b

    # 현재 상황과 가중치를 고려하여 제일 좋은 put을 반환하는 Wrapper 함수
    def calculate_best_put(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], float]:
        best_rule, max_weight = None, -1.0
        
        all_rules = list(DiceRule)
        current_numbers = self.get_importance_of_numbers(dice, state)

        if DEBUG_MODE: print(f"\n{self.round}R, dice: {sorted(dice)}, importance: {current_numbers}", file=sys.stderr) # 디버깅

        # 모든 규칙에 대해 점수를 계산
        for rule in all_rules:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))

                if score <= 0:
                    continue
                
                basic_score = sum(s for i, s in enumerate(state.rule_score) if s and i <= 5)
                # Bonus를 획득하지 않은 경우 기본 규칙에 importance를 사용
                if basic_score < 63000 and rule.value <= 5:
                    # 기본 규칙 중 점수 미등록한 개수
                    remaining_basic = sum(1 for i in range(6) if state.rule_score[i] is None) 
        
                    if rule.value <= 3: # 1, 2, 3과 같은 낮은 숫자
                        importance = 1.5 + (0.1) * remaining_basic
                    else: # 4, 5, 6과 같은 높은 숫자
                        importance = 2.5  + (0.1) * remaining_basic
                else:
                    importance = 1
                
                tmp_utility = self._get_utility_of_rules(score, rule, dice, state)
                utility = (tmp_utility * importance)
                if DEBUG_MODE: print(f"{self.round}R, score: {score // 1000}, rule: {rule.name}, utility: {score / AVERAGE_SCORES.get(rule, 1)}, get_ut: {tmp_utility}, importance: {importance}, total_weight: {utility}", file=sys.stderr) # 디버깅

                if utility > max_weight:
                    max_weight, best_rule = utility, rule
        
        if max_weight != -1.0:
            if DEBUG_MODE: print(f"{self.round}R, calculate_best_put() END -> rule: {best_rule.name}, local_max_weight: {max_weight}", file=sys.stderr) # 디버깅
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
        if put.rule is None: return
        assert self.rule_score[put.rule.value] is None
        for d in put.dice:
            if d in self.dice: self.dice.remove(d)
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
            return 15000 if "1234" in unique_dice_str or "2345" in unique_dice_str or "3456" in unique_dice_str else 0
        if rule == DiceRule.LARGE_STRAIGHT: 
            return 30000 if "12345" in unique_dice_str or "23456" in unique_dice_str else 0
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
                    dice_a[i] = int(c)  # 문자를 숫자로 변환
                for i, c in enumerate(str_b):
                    dice_b[i] = int(c)  # 문자를 숫자로 변환

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