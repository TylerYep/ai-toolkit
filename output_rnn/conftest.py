''' conftest.py '''
# from typing import Tuple
# from collections import Counter
# import random
# import pytest

# from src import const
# from fixtures import *

# @pytest.fixture(autouse=True)
# def reset_const():
#     const.logger.setLevel(const.TRACE)
#     const.ROLES = ('Insomniac', 'Villager', 'Robber', 'Villager', 'Drunk', 'Wolf', 'Wolf', 'Seer',
#                    'Tanner', 'Mason', 'Minion', 'Troublemaker', 'Villager', 'Mason', 'Hunter')
#     const.ROLE_SET = set(const.ROLES)
#     const.ROLE_COUNTS = dict(Counter(const.ROLES))
#     const.NUM_ROLES = len(const.ROLES)
#     const.NUM_PLAYERS = 12
#     const.NUM_CENTER = 3
#     const.CENTER_SEER_PROB = 0.9
#     const.VILLAGE_ROLES = {'Villager', 'Mason', 'Seer', 'Robber', 'Troublemaker',
#                            'Drunk', 'Insomniac', 'Hunter'}
#     const.EVIL_ROLES = {'Tanner', 'Wolf', 'Minion'}
#     const.USE_VOTING = True
#     const.USE_REG_WOLF = False
#     const.USE_EXPECTIMAX_WOLF = False
#     random.seed(0)


# def debug_spacing_issues(captured: str, expected: str):
#     ''' Helper method for debugging print differences. '''
#     print(len(captured), len(expected))
#     for i, captured_char in enumerate(captured):
#         if captured_char != expected[i]:
#             print("INCORRECT: ", i, captured_char, "vs", expected[i])
#         else:
#             print(" " * 10, i, captured_char, "vs", expected[i])
