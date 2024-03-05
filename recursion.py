def change(amount, coins, memo=None):
    if memo is None:
        memo = {}
    if amount == 0:
        return 0
    if amount in memo:
        return memo[amount]
    res = float('inf')
    for coin in coins:
        if coin <= amount:
            sub_res = change(amount - coin, coins, memo)
            if sub_res != float('inf') and sub_res + 1 < res:
                res = sub_res + 1
    memo[amount] = res
    return res


def giveChange(amount, coins):
    if amount == 0:
        return 0, []
    res = float('inf')
    res_coins = []
    for coin in coins:
        if coin <= amount:
            sub_res, sub_res_coins = giveChange(amount - coin, coins)
            if sub_res != float('inf') and sub_res + 1 < res:
                res = sub_res + 1
                res_coins = sub_res_coins + [coin]
    return res, res_coins

# Example usage
if __name__ == "__main__":
    amount = 11
    coins = [1, 2, 5]
    print(f"Minimum coins needed: {change(amount, coins)}")
    min_coins, coins_used = giveChange(amount, coins)
    print(f"Minimum coins needed: {min_coins}, Coins used: {coins_used}")
