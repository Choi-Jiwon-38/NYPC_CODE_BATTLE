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
# - 점수 우선 순위 -
# 1. 야추
# 2. FOUR, FIVE, SIX 적어도 3개이상
# 3. 숫자가 큰 Full House, Four of kind
# 4. Large Straight, Small Straight
# 5. 숫자가 작은 Full House, Four of kind
# 6. ONE, TWO ,THREE (보통 마땅히 않을 때 0으로 버릴 가능성이 높음)

W_YACHT = 3.0
W_LARGE_STRAIGHT = 2.3
W_SMALL_STRAIGHT = 1.8
W_DEMOTION = 0.3
W_HIGH_PROMOTION = 2.5
W_LOW_PROMOTION = 1.5
W_NICE_CHOICE = 1.7
W_BAD_CHOICE = 0.5
W_BASIC = 0.3
W_SAVING = 0.7

# 숫자의 중요도
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

LOW_UTILITY = 0.01        # 효율성이 이 이하이면 SACRIFICE
SCORE_GOOD_CHOICE = 22    # W_NICE_CHOICE를 적용할 임계 점수
SCORE_HIGH_FULLHOUSE = 18 # Full House가 좋은 선택일지 정하는 임계 점수
SCORE_HIGH_FOK = 15       # Four of Kind가 좋은 선택일지 정하는 임계 점수
NR_END_GAME = 2           # 게임 후반부인지 판단할 남은 Rule 개수의 임계값
NR_BUFFERING_1 = 1        # 1을 버퍼로 남겨두기 위해 필요한 Rule 개수의 임계값

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

    # ================================ [필수 구현] ================================

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        # 각 주사위 묶음의 최대 기대 효율(utility)을 계산
        _, score, utility_a = self.calculate_best_put(dice_a, self.my_state)
        _, _, utility_b = self.calculate_best_put(dice_b, self.my_state)

        # 더 높은 효율을 가진 그룹에 입찰
        group = "A" if utility_a > utility_b else "B"
        
        # 보수적인 입찰 금액 산정
        score_diff = self.my_state.get_total_score() - self.opp_state.get_total_score()

        # 만약 5000점 이상 이기고 있다면, 위험을 감수하지 않고 0을 베팅하여 리드를 지킴
        if score_diff > 5000:
            return Bid(group, 0)
        else:
            # 상대방 히스토리 기반 베팅 금액 계산
            if self.opp_bid_history:
                # 상대방 베팅 히스토리 분석
                max_opp_bid = 1 if max(self.opp_bid_history) == 0 else max(self.opp_bid_history)
                sorted_bids = sorted(self.opp_bid_history, reverse=True)
                top_3_avg = 1 if sum(sorted_bids[:3]) / min(3, len(sorted_bids)) == 0 else sum(sorted_bids[:3]) / min(3, len(sorted_bids))
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
            else:
                amount = 0
        
        return Bid(group, amount)
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
            best_rule, _, utility = self.calculate_best_put(dice_list, self.my_state)
            if utility > max_utility:
                max_utility = utility
                best_put = [DicePut(best_rule, dice_list)]
            elif utility == max_utility:
                best_put.append(DicePut(best_rule, dice_list))

        if len(best_put) == 0 or max_utility <= LOW_UTILITY:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            self.update_importance_of_numbers()
            # 중요도가 낮은 순으로 5개 선택
            dice_to_sacrifice = sorted(self.my_state.dice, key=lambda d: W_NUMBERS[d - 1])[:5]
            return DicePut(rule_to_sacrifice, dice_to_sacrifice)

        if len(best_put) == 1:
            return best_put[0]
        
        # 여러 후보 중 중요도 합이 가장 낮은 것을 선택
        self.update_importance_of_numbers()
        def importance_sum(dice_list):
            return sum(W_NUMBERS[val - 1] for val in dice_list)
        best_put.sort(key=lambda put: (importance_sum(put.dice), sum(put.dice)))
        return best_put[0]
    # ============================== [필수 구현 끝] ==============================

    # W_NUMBERS를 계산하는 내부 함수
    def update_importance_of_numbers(self) -> None:
        _rule_score = self.my_state.rule_score
        _dice_count = [self.my_state.dice.count(i) for i in range(1, 7)]

        # 현재 보유 중인 숫자가 많은 경우에는 최대한 사용하지 않도록 함
        for num in range(6):
            W_NUMBERS[num] *= (1 + 0.1 * _dice_count[num])

        # 기본 점수 규칙(ONE ~ SIX)을 만족하지 못한 경우에는 해당 숫자의 중요도를 올림.
        for num in range(6):
            if _rule_score[num] is None:
                W_NUMBERS[num] *= W_IMPORTANT
            else:
                W_NUMBERS[num] *= W_NOT_IMPORTANT

    # utility를 계산하는 내부 함수
    def get_utility_of_rules(self, score: int, rule: DiceRule) -> float:
        pass

    # 가중치를 이용해 제일 좋은 put을 반환하는 Wrapper 함수
    def calculate_best_put(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], int, float]:
        best_rule, best_score, max_utility = None, -1, -1.0
        
        all_rules = list(DiceRule)
        
        for rule in all_rules:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))

                if score <= 0:
                    continue

                # 1. 기본 효율성(Utility) 계산
                utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0
                
                # 2. 우선순위에 따른 전략적 가중치 적용
                # --- 기본 숫자 규칙 (ONE ~ SIX) ---
                if rule.value <= 5:
                    dice_number = rule.value + 1
                    count_of_number = dice.count(dice_number)
                    
                    # 우선순위 2번: 4, 5, 6이 3개 이상일 때만 높은 가치를 부여
                    # FOUR 규칙은 4개 이상일 때만 높은 가중치 적용
                    if dice_number == 4:
                        if count_of_number >= 4:
                            utility *= W_HIGH_PROMOTION
                        else:
                            utility *= W_DEMOTION
                    
                    # FIVE와 SIX 규칙은 3개 이상일 때만 높은 가중치 적용
                    elif dice_number in [5, 6]:
                        if count_of_number >= 3:
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
                    elif rule == DiceRule.CHOICE:
                        if dice_sum >= SCORE_GOOD_CHOICE:
                            utility *= W_NICE_CHOICE
                        else:
                            utility *= W_BAD_CHOICE

                # 3. 전체 게임 상황을 고려한 추가 가중치 적용
                utility = self._apply_strategy(rule, score, dice, utility, state)
                
                # 4. 미래 가치를 위해 현재 규칙을 아껴두는 'Saving' 전략
                # TODO
                # 이거 필요없는거 아님???
                remaining_dice = [d for d in state.dice if d not in dice]
                if len(remaining_dice) >= 5: # 남아있는 주사위가 5개 이상일 때 (1라운드, 13라운드 제외)
                    from collections import Counter
                    counts = Counter(remaining_dice)
                    # 남은 주사위에 같은 숫자가 4개 이상이면, 다음 턴 Yacht를 기대
                    if counts.most_common(1)[0][1] >= 4:
                        if rule == DiceRule.YACHT:
                            utility *= W_SAVING
                    
                    # 남은 주사위가 스트레이트를 만들기 좋다면, 스트레이트 규칙을 아껴둠
                    # 남은 주사위 중 다른 숫자가 4개 이상이고, 그 중 3개 이상이 연속되는지 확인
                    unique_remaining = sorted(list(set(remaining_dice)))
                    if len(unique_remaining) >= 3:
                        has_potential = False
                        # 3칸 연속 (예: 1,2,3 또는 2,3,4)이 있는지 확인
                        for i in range(len(unique_remaining) - 2):
                            if unique_remaining[i+1] == unique_remaining[i] + 1 and \
                                unique_remaining[i+2] == unique_remaining[i] + 2:
                                has_potential = True
                                break
                            
                        if has_potential and rule in [DiceRule.SMALL_STRAIGHT, DiceRule.LARGE_STRAIGHT]:
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