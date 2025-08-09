from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations

import sys


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
    DiceRule.ONE: 1880, DiceRule.TWO: 5280, DiceRule.THREE: 8570,
    DiceRule.FOUR: 12160, DiceRule.FIVE: 15690, DiceRule.SIX: 19190,
    DiceRule.CHOICE: 22000, DiceRule.FOUR_OF_A_KIND: 13100, DiceRule.FULL_HOUSE: 22590,
    DiceRule.SMALL_STRAIGHT: 15000, DiceRule.LARGE_STRAIGHT: 30000, DiceRule.YACHT: 17000,
}

# 점수를 버려야 할 때의 우선순위 (사용자 전략 기반)
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

# 가중치 상수들
# 가중치 조정할 때 숫자를 변경하여 조절할 것!
W_YACHT = 3.0
W_LARGE_STRAIGHT = 2.5
W_SMALL_STRAIGHT = 1.8
W_DEMOTION = 0.3
W_HIGH_PROMOTION = 2.0
W_LOW_PROMOTION = 1.3
W_NICE_CHOICE = 1.7
W_BAD_CHOICE = 0.8
W_BASIC = 0.6
W_SAVING = 0.8

W_IMPORTANT = 1.2
W_NOT_IMPORTANT = 0.8
W_NUMBERS = [
    1,     # 1
    1.2,   # 2
    1.4,   # 3
    2,     # 4
    2.2,   # 5
    2.4    # 6
] # 높은 숫자일수록 기본적으로 중요도가 높음

LOW_UTILITY = 0.01 # 효율성이 이 이하이면 SACRIFICE
NR_END_GAME = 3    # 게임 후반부인지 판단할 남은 Rule 개수의 임계값
NR_BUFFERING_1 = 4 # 1을 버퍼로 남겨두기 위해 필요한 Rule 개수의 임계값
NR_SAVING = 4           # 몇 개나 일치할 때 W_SAVING을 곱할지 정하는 임계값

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

    # ================================ [필수 구현] ================================

    # 1450점의 베팅 전략
    # 지금은 아직 점수 선택을 먼저 수정해야 하므로
    # 베팅은 기본 베팅으로 적용!
    def _calculate_bid_amount(self, utility_a: float, utility_b: float, score_diff: int) -> int:
        ### 상대방 히스토리 기반 베팅 금액 계산 (전략적 개선) ###
        
        # 기본 베팅 금액
        utility_diff = abs(utility_a - utility_b)
        base_amount = int(utility_diff * 3000)
        
        # 상대방 히스토리가 없으면 기본 금액 사용
        if not self.opp_bid_history:
            return min(base_amount, 2000)
        
        # 상대방의 최근 베팅 패턴 분석
        recent_bids = self.opp_bid_history[-5:]  # 최근 5개
        avg_bid = sum(recent_bids) / len(recent_bids)
        max_recent_bid = max(recent_bids)
        min_recent_bid = min(recent_bids)
        
        # 상대방 예상 베팅액 계산
        if len(recent_bids) >= 3:
            # 최근 트렌드 분석
            recent_trend = recent_bids[-1] - recent_bids[-3]  # 최근 3라운드 트렌드
            if recent_trend > 0:  # 베팅액이 증가하는 추세
                expected_bid = max_recent_bid + int(recent_trend * 0.5)
            else:  # 베팅액이 감소하거나 유지
                expected_bid = avg_bid
        else:
            expected_bid = avg_bid
        
        # 전략적 베팅액 결정
        if score_diff < 0:  # 지고 있을 때 - 더 적극적으로
            # 상대방 예상 베팅액보다 최소 10% 더 높게 베팅
            strategic_amount = int(expected_bid * 1.1)
            # 뒤처진 점수 고려
            max_bid = abs(score_diff) + 2000
            amount = min(strategic_amount, max_bid)
        else:  # 이기고 있을 때 - 안정적으로
            # 상대방 예상 베팅액과 비슷하게 베팅
            strategic_amount = int(expected_bid * 0.9)
            amount = min(strategic_amount, 3000)
        
        # 최소 베팅액 보장 (상대방이 우리를 이용하는 것 방지)
        min_bid = max(500, int(avg_bid * 0.8))
        amount = max(amount, min_bid)
        
        return max(0, int(amount))
    
    # [개선됨] 안정적으로 점수를 관리하는 '보수적인 입찰 전략'
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        # 각 주사위 묶음의 최대 기대 효율(utility)을 계산
        _, _, utility_a = self._calculate_best_put_for_dice(dice_a, self.my_state)
        _, _, utility_b = self._calculate_best_put_for_dice(dice_b, self.my_state)

        # 더 높은 효율을 가진 그룹에 입찰
        group = "A" if utility_a > utility_b else "B"
        
        # 보수적인 입찰 금액 산정
        score_diff = self.my_state.get_total_score() - self.opp_state.get_total_score()

        # 만약 5000점 이상 이기고 있다면, 위험을 감수하지 않고 0을 베팅하여 리드를 지킴
        if score_diff > 5000:
            return Bid(group, 0)
        
        # 상대방 히스토리 기반 베팅 금액 계산
        amount = self._calculate_bid_amount(utility_a, utility_b, score_diff)
        
        return Bid(group, max(0, min(100000, int(amount))))

    def get_dynamic_sacrifice_priority(self) -> List[DiceRule]:
        ### 현재 상황에 따른 동적 우선순위 계산 (개선됨) ###
        priorities = []
        
        # 기본 점수 규칙들 (ONE~SIX)
        basic_rules = [DiceRule.ONE, DiceRule.TWO, DiceRule.THREE, 
                       DiceRule.FOUR, DiceRule.FIVE, DiceRule.SIX]
        
        # 현재 기본 점수 계산
        current_basic = sum(s for i, s in enumerate(self.my_state.rule_score) if s and i <= 5)
        
        # 보너스 점수(35000)를 고려한 우선순위 (강화됨)
        if current_basic < 63000:
            # 보너스를 얻을 수 있다면 기본 규칙을 절대 버리지 않음
            available_basic = [r for r in basic_rules if self.my_state.rule_score[r.value] is None]
            if available_basic:
                # 기본 규칙을 우선적으로 보존
                priorities.extend(available_basic)
            
            # 조합 규칙들 (높은 점수 순, 보너스 획득 전에는 신중하게)
            combination_rules = [
                DiceRule.SMALL_STRAIGHT, DiceRule.FOUR_OF_A_KIND, 
                DiceRule.FULL_HOUSE, DiceRule.LARGE_STRAIGHT, DiceRule.YACHT
            ]
            priorities.extend([r for r in combination_rules if self.my_state.rule_score[r.value] is None])
        else:
            # 보너스를 이미 얻었다면 기본 규칙을 먼저 버림
            priorities.extend([r for r in basic_rules if self.my_state.rule_score[r.value] is None])
            
            # 조합 규칙들 (높은 점수 순)
            combination_rules = [
                DiceRule.SMALL_STRAIGHT, DiceRule.FOUR_OF_A_KIND, 
                DiceRule.FULL_HOUSE, DiceRule.LARGE_STRAIGHT, DiceRule.YACHT
            ]
            priorities.extend([r for r in combination_rules if self.my_state.rule_score[r.value] is None])
        
        return priorities
    """

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        # 고정된 입찰 전략: 무조건 A를 선택하고 0원으로 베팅
        return Bid("A", 0)
    """
    def calculate_put(self) -> DicePut:
        best_put = []
        max_utility = -1.0
        
        dice_pool = self.my_state.dice
        num_to_pick = 5 if len(dice_pool) >= 5 else len(dice_pool)
        if num_to_pick == 0:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            return DicePut(rule_to_sacrifice, [])

        for dice_combination in combinations(dice_pool, num_to_pick):
            dice_list = list(dice_combination)
            best_rule, _, utility = self._calculate_best_put_for_dice(dice_list, self.my_state)
            if utility > max_utility:
                max_utility = utility
                best_put = [DicePut(best_rule, dice_list)]
            elif utility == max_utility:
                best_put.append(DicePut(best_rule, dice_list))

        if len(best_put) == 0 or max_utility <= LOW_UTILITY:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            importance = self.get_importance_of_numbers()
            # 중요도가 낮은 순으로 5개 선택
            dice_to_sacrifice = sorted(self.my_state.dice, key=lambda d: importance[d - 1])[:5]
            return DicePut(rule_to_sacrifice, dice_to_sacrifice)

        if len(best_put) == 1:
            return best_put[0]
        
        # 여러 후보 중 중요도 합이 가장 낮은 것을 선택
        importance = self.get_importance_of_numbers()
        def importance_sum(dice_list):
            return sum(importance[val - 1] for val in dice_list)
        best_put.sort(key=lambda put: (importance_sum(put.dice), sum(put.dice)))
        return best_put[0]
    # ============================== [필수 구현 끝] ==============================

    def get_importance_of_numbers(self) -> List[int]:
        _rule_score = self.my_state.rule_score
        _dice_count = [self.my_state.dice.count(i) for i in range(6)]

        # 현재 보유 중인 숫자가 많은 경우에는 최대한 사용하지 않도록 함
        for num in range(6):
            W_NUMBERS[num] *= (1 + 0.1 * _dice_count[num])

        # 기본 점수 규칙(ONE ~ SIX)을 만족하지 못한 경우에는 해당 숫자의 중요도를 올림.
        for num in range(6):
            if _rule_score[num] is None:
                W_NUMBERS[num] *= W_IMPORTANT
            else:
                W_NUMBERS[num] *= W_NOT_IMPORTANT

        return W_NUMBERS

    def _calculate_best_put_for_dice(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], int, float]:
        best_rule, best_score, max_utility = None, -1, -1.0
        
        # 모든 규칙을 점수 순으로 정렬하여 시도
        all_rules = list(DiceRule)
        
        for rule in all_rules:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))
                if score > 0:  # 점수가 있는 경우만 고려
                    utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0
                    utility = self._apply_strategy(rule, score, dice, utility, state)
                    
                    # 조합 규칙 우선 가중치
                    if rule == DiceRule.YACHT:
                        utility *= W_YACHT  # YACHT는 최우선
                    elif rule == DiceRule.LARGE_STRAIGHT:
                        utility *= W_LARGE_STRAIGHT  # Large Straight는 매우 우선
                    elif rule == DiceRule.SMALL_STRAIGHT:
                        utility *= W_SMALL_STRAIGHT  # Small Straight도 우선
                    elif rule in [DiceRule.FOUR_OF_A_KIND, DiceRule.FULL_HOUSE]:
                        # 낮은 숫자 조합은 페널티
                        dice_sum = sum(dice)
                        if dice_sum < 15:
                            utility *= W_DEMOTION  # 낮은 합계면 큰 페널티
                        else:
                            utility *= W_HIGH_PROMOTION  # 높은 합계면 보너스
                    elif rule == DiceRule.CHOICE:
                        dice_sum = sum(dice)
                        if dice_sum >= 20:
                            utility *= W_NICE_CHOICE  # 높은 합계면 큰 보너스
                        else:
                            utility *= W_BAD_CHOICE  # 기본 보너스
                    elif rule.value <= 5: # 기본 규칙 (ONE~SIX)
                        dice_number = rule.value + 1
                        # 4는 4개 / 5, 6은 3개 이상일 때 높은 가중치를 부여합니다.
                        if dice_number in [4]:
                            count_of_number = dice.count(dice_number)
                            if count_of_number >= 4:
                                utility *= W_HIGH_PROMOTION
                        elif dice_number in [5, 6]:
                            count_of_number = dice.count(dice_number)
                            if count_of_number >= 3:
                                utility *= W_HIGH_PROMOTION
                                
                        # 일반적인 높은 숫자 선호 전략을 사용합니다.
                        utility *= (1 + dice_number * W_BASIC)

                    # === 미래 가치 보존 전략 추가 ===
                    # 남은 주사위들로 더 좋은 점수를 기대할 수 있는 규칙은 현재 가치를 약간 낮춰 아껴둠
                    remaining_dice = [d for d in state.dice if d not in dice]
                    if len(remaining_dice) >= 4:
                        # 남은 주사위에 같은 숫자가 4개 이상이면, 다음 턴 Yacht/Four-of-a-kind를 기대
                        from collections import Counter
                        counts = Counter(remaining_dice)
                        if counts.most_common(1)[0][1] >= NR_SAVING:
                            if rule in [DiceRule.YACHT, DiceRule.FOUR_OF_A_KIND]:
                                utility *= W_SAVING # 현재 이 규칙을 사용하는 것의 가치를 감소시켜 아껴둠

                        # 남은 주사위가 스트레이트를 만들기 좋다면, 스트레이트 규칙을 아껴둠
                        unique_remaining = sorted(list(set(remaining_dice)))
                        if len(unique_remaining) >= NR_SAVING: # Small Straight 가능성
                            if rule == DiceRule.SMALL_STRAIGHT:
                                utility *= W_SAVING
                    
                    if utility > max_utility:
                        max_utility, best_rule, best_score = utility, rule, score
        
        return best_rule, best_score, max_utility

    def _apply_strategy(self, rule: DiceRule, score: int, dice: List[int], utility: float, state: 'GameState') -> float:
        """사용자 전략을 적용하여 utility 조정"""
        
        # 1. 보너스 점수 전략 (63점 달성 시 35점 보너스)
        if rule.value <= 5:  # 기본 규칙 (ONE~SIX)
            basic_score = sum(s for i, s in enumerate(state.rule_score) if s and i <= 5)
            if basic_score < 63000 and basic_score + score >= 63000:
                utility *= W_YACHT  # 보너스 점수 가중치 증가 (야추급)
        
        # 2. 게임 진행 상황 고려
        remaining_rules = sum(1 for s in state.rule_score if s is None)
        if remaining_rules <= NR_END_GAME:  # 게임 후반부
            # 높은 점수 조합 우선
            if rule in [DiceRule.YACHT, DiceRule.LARGE_STRAIGHT, DiceRule.SMALL_STRAIGHT]:
                utility *= W_HIGH_PROMOTION
            # 낮은 점수라도 확실한 점수 확보
            elif score > 0:
                utility *= W_LOW_PROMOTION
        
        # 3. 기회 활용 (찬스나 낮은 족보도 상황에 따라 활용)
        if rule == DiceRule.CHOICE:
            dice_sum = sum(dice)
            if dice_sum >= 18:  # 높은 합계면 더 큰 보너스
                utility *= W_NICE_CHOICE
            elif dice_sum >= 12:  # 중간 합계면 기본 보너스
                utility *= W_LOW_PROMOTION
        
        # 4. 변수 대응 (유연한 전략)
        # 1은 버퍼로 사용하되, 다른 선택이 없으면 활용
        if rule == DiceRule.ONE:
            other_options = sum(1 for r in DiceRule if r != rule and state.rule_score[r.value] is None)
            if other_options > NR_BUFFERING_1:  # 다른 선택이 많으면 1은 페널티
                utility *= W_DEMOTION
            else:  # 다른 선택이 적으면 1도 활용
                utility *= W_LOW_PROMOTION
        
        return utility

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