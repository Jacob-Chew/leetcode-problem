import copy
import time


class ListNode():
    def __init__(self, val):
        self.val = val
        self.next = None


def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    最长公共子序列
    递归版本，边界条件是索引到-1则返回0，否则根据转移条件递归求解：
    如果当前字符相同，则返回前一个子序列最大值+1，
    否则在二者中选一个最大值。
    """
    if text2 is None or text1 is None:
        return 0

    def op(i, j):
        if i == -1 or j == -1:
            return 0
        elif text1[i] == text2[j]:
            return op(i-1, j-1)+1
        else:
            return max(op(i-1, j), op(i, j-1))

    return op(len(text1), len(text2))


def coinChange(coins, amount):
    # def helper(n):
    #     if n == 0:
    #         return 0
    #     elif n < 0:
    #         return -1
    #     res = float('inf')
    #     for coin in coins:
    #         if helper(n - coin) == -1:
    #             continue
    #         res = min(res, helper(n - coin)+1)
    #     return res if res != float('inf') else -1
    # return helper(amount)
    # 动态规划的迭代版本
    dp = [amount+1] * (amount+1)
    # base case
    dp[0] = 0
    for i in range(1, amount+1):
        for coin in coins:
            if i-coin < 0:
                continue
            dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount]


def permute( nums):
    """
    全排列问题  只适用于nums中元素都不同
    """
    result = []
    route = {}

    def track(nums, route):
        if len(route) == len(nums):
            result.append(list(route.keys()))
            return
        for val in nums:
            if val in route:
                continue
            route[val] = 1
            track(nums, route)
            del route[val]
    track(nums, route)
    return result


def print_all_subsquence(s):
    """
    打印字符串的所有子序列
    """
    # 第一种方法是递归
    res = []
    def all_subsquence(s, i, tmp):
        if i == len(s):
            res.append(tmp)
            return
        all_subsquence(s, i+1, tmp)       # 回溯 当前位置不加入该对应字符
        all_subsquence(s, i+1, tmp+s[i])  # 当前位置加入对应字符
        return
    all_subsquence(s, 0, "")

    # 第二种方法，迭代
    def iteration(s):
        ln = len(s)
        # 总的2^n 种情况，即对于s每一位，都有 要或者不要 两种情况 假设s长度为四，对应二进制 0000 0001 0010 0011 ... 1表示该位要0表示不要
        nums = 1 << ln
        for i in range(nums):  # 这里即是遍历 0000 ~ 1111
            tmp = ''
            for j in range(ln):
                if 1 << j & i:   # j 则是遍历i的所有位  看是0还是1，是1 则加入当前s[j]
                    tmp += s[j]
            res.append(tmp)
        return res
    return res




def solveNQueens( n: int):
    """
    N皇后问题求解  回溯法
    :param n:
    :return:
    """
    board = [['.']*n for _ in range(n)]   # 建立空棋盘
    result = []  # 储存正确路径

    def isValid(board, row, col):  # 给到Q的坐标，因为每行元素会逐个判断，所以只需判断每一列、两个斜上列是否满足条件
        for i in range(row):
            his = board[i].index('Q') # 之前行皇后所在的列数
            if his == col or abs(his-col) == abs(i-row):
                return False

        # for i in range(n):   # 判断第col列是否有皇后
        #     if board[i][col] == 'Q':
        #         return False
        # i, j = row, col
        # while i >= 0 and j < n:  # 判断右上斜列是否有皇后
        #     if board[i][j] == 'Q':
        #         return False
        #     i -= 1
        #     j += 1
        # i, j = row, col
        # while i >= 0 and j >= 0:  # 判断左上斜列是否有皇后
        #     if board[i][j] == 'Q':
        #         return False
        #     i -= 1
        #     j -= 1
        return True

    def backtrack(board, row):
        if row == n:
            res = []
            for i in range(n):
                temp = ''.join(board[i])
                res.append(temp)
            result.append(res)
            return
        for col in range(n):
            if not isValid(board, row, col):
                continue
            board[row][col] = 'Q'
            backtrack(board, row+1)
            board[row][col] = '.'

        return result

    return backtrack(board, 0)


def maxProfit_2(prices):
    """
    允许最多两次交易，来获得最大股票利润
    采用DP表方案  dp[i][k][0] 表示第i天，此时允许交易k次，手上没有股票  此时的最大利润
                而dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1]+price[i])
                  dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0]-price[i])
    本质是遍历所有方案
    """
    # 当k=1时

    # dp = [[]]
    # dp[0][0] = 0
    # dp[0][1] = -prices[0]    # base case
    # for i in range(1, len(prices)):
    #     # 转移方程
    #     dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
    #     dp[i][1] = max(dp[i-1][1], -prices[i])
    # return dp[-1][0]

    # 当k=2时
    dp = [[[]]]
    dp[0][2][0] = 0                                     # a_20 = 0
    dp[0][2][1] = -float('inf')                         # a_21 = -float('inf')
    dp[0][1][0] = 0             # base case             # a_10 = 0
    dp[0][1][1] = -float('inf')                         # a_11 = -float('inf')
    for i in range(1, len(prices)):
        dp[i][2][0] = max(dp[i-1][2][0], dp[i-1][2][1]+prices[i])   # a_20 = max(a_20, a_21+price[i])
        dp[i][2][1] = max(dp[i-1][2][1], dp[i-1][1][0]-prices[i])   # a_21 = max(a_21, a_10-price[i])
        dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1]+prices[i])   # a_10 = max(a_10, a_11+price[i])
        dp[i][1][1] = max(dp[i-1][1][1], -prices[i])                # a_11 = max(a_11, -price[i])
    return dp[-1][2][0]


def fast_sort(nums, l, r):
    def partition(nums, l, r):
        flag = nums[l]
        lo, hi = l, r
        while lo < hi:
            while lo < hi and nums[hi] >= flag:
                hi -= 1
            nums[lo] = nums[hi]
            while lo < hi and nums[lo] < flag:
                lo += 1
            nums[hi] = nums[lo]
        nums[lo] = flag
        return lo
    if r <= l:
        return
    mid = partition(nums, l, r)
    fast_sort(nums, l, mid-1)
    fast_sort(nums, mid+1, r)


def simple_quick_sort(nums, l, r):
    if r <= l:
        return
    flag = nums[l]
    lo, hi = l, r
    while lo < hi:
        while lo < hi and nums[hi] >= flag:
            hi -= 1
        nums[lo] = nums[hi]
        while lo < hi and nums[lo] < flag:
            lo += 1
        nums[hi] = nums[lo]
    nums[lo] = flag
    simple_quick_sort(nums, l, lo - 1)
    simple_quick_sort(nums, lo + 1, r)


def findMode(self, root):
    most = 0
    before = None
    cnt = 0
    result = []
    def inorder(root):
        nonlocal most, before, cnt, result
        if not root:
            return
        inorder(root.left)
        val = root.val
        if val == before:
            cnt += 1
        else:
            before = val
            cnt = 0
        if cnt == most:
            result.append(val)
        elif cnt > most:
            result = [val]
            cnt = 0
        inorder(root.right)
        return result
    return inorder(root)


def largestPerimeter(A) -> int:
    """
    A中是一系列正整数代表长度，找出所有能组合成三角形的长度组合中，和最大的数字组合，返回这个最大和；如果都不符合三角形条件，返回0
    """
    if len(A) < 3:
        return 0
    A.sort(reverse=True)
    for i in range(0, len(A)-2):
        if A[i] < A[i+1]+A[i+2]:
            return A[i] + A[i+1] + A[i+2]
    return 0

"""
递归的方式翻转链表
"""
def reverse(head):
    # 因为是翻转子链表，考虑末尾链表已经翻转好 1 -> 2 <- 3 <- 4
    if not head or not head.next: # 如果是空表或者单链表，直接返回head
        return head
    tmp = reverse(head.next)
    head.next.next, head.next = head, None
    return tmp  # 一直返回的是翻转后的头结点，即原来顺序的最后一个节点

"""
迭代方式翻转链表
"""
def reverse_2(head):
    if not head:
        return head
    dum = ListNode(None)
    while head:
        dum.next, head.next, head = head, dum.next, head.next

    return dum.next



def sortColors(self, nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    count=0
    for i in range(len(nums)):
        i=i-count
        print('change ',end='')
        #print(nums)
        print(nums[i])
        if nums[i]==0:
            nums.pop(i)
            nums=[0]+nums
        elif nums[i]==2:
            nums.pop(i)
            nums.append(2)
            count=count+1
    print(nums)


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList) -> int:
        """
        单词接龙， 最短单词转换路径，每次只能变换单词中的一个字母  从beginword 转换到  endword
        :return:
        """
        if endWord not in set(wordList):
            return 0
        distance_table = self.get_distance(beginWord, wordList)
        nexts = self.get_next(beginWord, wordList)
        paths = []
        solution = []
        self.get_path(beginWord, endWord, distance_table, paths, solution, nexts)
        path_len = []
        if not paths:
            return 0
        for path in paths:
            path_len.append(len(path))
        return min(path_len)

    def get_path(self, cur_word, end_word, distances, res, solution, nexts):
        """
        深度优先遍历  获得begin到end的路径  递归
        :return:
        """
        solution.append(cur_word)
        if cur_word == end_word:
            res.append(solution.copy())
        else:
            for wrd in nexts[cur_word]:
                if distances[wrd] - distances[cur_word] == 1:
                    self.get_path(wrd, end_word, distances, res, solution, nexts)
        solution.pop()

    def get_next(self, begin_word, word_list):
        """
        获得单词的所有邻居（距离为1）, 即返回值是一个字典，key是单词，value是含距key为1的所有单词的列表
        :param begin_word:
        :param word_set:
        :return:  nexts
        """
        word_set = set(word_list)
        len_word = len(begin_word)
        nexts = {}
        word_list.append(begin_word)
        for wrd in word_list:
            nexts[wrd] = []

            for i in range(len_word):
                for j in range(26):
                    new_wrd = list(wrd)
                    new_wrd[i] = chr(ord('a')+j)
                    new_wrd = ''.join(new_wrd)
                    if wrd != new_wrd:
                        if new_wrd in word_set:
                            nexts[wrd].append(new_wrd)
        return nexts

    def get_distance(self, begin, word_list):
        """
        返回一个字典，key是word list中的所有单词，value是key和begin word的距离
        :param begin:
        :return:
        """
        nexts = self.get_next(begin, word_list)
        distances = {begin: 0}
        is_visited = set()
        is_visited.add(begin)
        queue = [begin]
        while queue:
            tpt = queue.pop(0)
            for nxt_wrd in nexts[tpt]:
                if nxt_wrd not in is_visited:
                    is_visited.add(nxt_wrd)
                    queue.append(nxt_wrd)
                    distances[nxt_wrd] = distances[tpt] + 1
        return distances


class Solution_2:
    def movingCount(self, m: int, n: int, k: int) -> int:
        counts = 0
        visited = [[0] * n for _ in range(m)]
        return self.count(m, n, 0, 0, k, visited)

    def count(self, m, n, x_indx, y_indx, k, visited):
        num = 0
        if self.can_move(m, n, x_indx, y_indx, k, visited):
            visited[x_indx][y_indx] = 1
            num = 1 + self.count(m, n, x_indx + 1, y_indx, k, visited) + self.count(m, n, x_indx - 1, y_indx, k, visited) + \
                  self.count(m, n, x_indx, y_indx + 1, k,visited) + self.count(m, n, x_indx, y_indx - 1, k, visited)
        return num

    def can_move(self, m, n, x_indx, y_indx, k, visited):
        if 0 <= x_indx < m  and 0 <= y_indx < n and not visited[x_indx][y_indx]:
            if self.get_sum(x_indx) + self.get_sum(y_indx) <= k:
                return True
        return False

    def get_sum(self, number):
        num_sum = 0
        while number:
            num_sum += number % 10
            number //= 10
        return num_sum


class Solution3:

    def partition(self, s: str):
        res = []
        solution = []
        self.find_solution(s, res, solution)
        return res

    def find_solution(self, strs, res, solution):
        if not strs:
            res.append(solution.copy())
        for i in range(1, len(strs)+1):
            if self.is_pal(strs[:i]):
                solution.append(strs[:i])
                self.find_solution(strs[i:], res, solution)
                solution.pop()

    def is_pal(self, str_s):
        return str_s == str_s[::-1]


def mergeNums(nums):
    """

    :param nums:
    :return: 合并k个有序数组  分治法，将nums中所有分为左右两组，再继续往下分，直到每组只有两个列表或者一个列表
    然后往上合并两个有序数组即可。 最小子问题就是合并两个有序数组
    """
    def merge2Nums(n1, n2):
        # 这里有两种方式 第一python的方法，直接在n1的中间insert  但是该方法比较偷懒；第二种方法用c++，n1的长度是n1+n2
        # 此时可以有两种方法 第一是建立一个空列表 双指针往里面放，第二是从n1的尾部开始放，也是双指针


        return n1

    def deal(nums, l, r):
        if l == r:
            return nums[l]
        elif r-l==1:
            return merge2Nums(nums[l], nums[r])
        mid = (l+r) // 2
        n1 = deal(nums, l, mid)
        n2 = deal(nums, mid+1, r)
        return merge2Nums(n1, n2)

    return deal(nums, 0, len(nums)-1)

# 数组的第k大元素
class Deal:
    def findKthLargest(self, nums, k: int) -> int:
        self.res = None
        def partion(nums, l, r):
            flag = nums[l]
            while l<r:
                while l<r and nums[r] <= flag:
                    r -= 1
                nums[l] = nums[r]
                while l<r and nums[l] >= flag:
                    l +=1
                nums[r] = nums[l]
            nums[l] = flag
            return l

        def quick_sort(nums, l, r):
            if l >= r:
                return
            indx = partion(nums, l, r)
            if indx ==k-1:
                self.res = nums[k-1]
                return
            elif indx<k-1:
                quick_sort(nums, indx+1, r)
            else:
                quick_sort(nums, l, indx-1)

        quick_sort(nums, 0, len(nums)-1)
        return self.res if self.res else nums[k-1]

if __name__ == '__main__':
    peach, deadline = map(int, input().split())
    times = list(map(int, input().split()))
    cnt = 0
    time_sum = 0


    def quick_sort(nums, l, r):
        if l >= r:
            return
        index = partion(nums, l, r)
        quick_sort(nums, l, index - 1)
        quick_sort(nums, index + 1, r)
        return nums


    def partion(nums, left, right):
        piv = nums[left]
        while left < right:
            while left < right and nums[right] >= piv:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] <= piv:
                left += 1
            nums[right] = nums[left]
        nums[left] = piv
        return left


    quick_sort(times, 0, peach - 1)
    for time in times:
        time_sum += time
        if time_sum < deadline:
            cnt += 1
        else:
            break
    print(cnt)

# 面试常考算法
# 1. 排序算法
# lc 215  数组的第k个最大大元素：快排 + 剪枝(可以不同排k不在的子数组)
# lc 253  or lintcode 919     会议室II
# 有序数组的平方
# 最佳见面地点 ：取出所有人的x，y坐标，分别排序 再取中位数 得到(xi, yi)即是最佳见面地点
# 摆动排序

# 2.###动态规划
"""
lc5 最长回文子串
问题：给你一个字符串 s，找到 s 中最长的回文子串。
解法：遍历每一个字符，以它为中心向两边展开
"""
class Solution_1:
    def longestPalindrome(self, s: str) -> str:
        # 中心往两边找
        def palin(s, l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1: r]

        res = ''
        for i in range(len(s)):
            s1 = palin(s, i, i)
            s2 = palin(s, i, i + 1)
            res = s1 if len(s1) > len(res) else res
            res = s2 if len(s2) > len(res) else res
        return res


"""
# lc91 解码方法
问题：一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：‘A'-> 1, 'B'->2, ... 要解码信息必须按照上述映射方法 注意不能将 06 解码为
'F' 因为06 和 6 不同
解法： dp[i]表示0-i的字符串有几种解码方法，那么dp[i+1]则 分几种情况nums[i]只能单个，只能和前一个组，或两个都行，或者0种解码方法
示例
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）
"""
class Solution_2:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0':
            return 0
        elif len(s) == 1:
            return 1
        a, b, c = 1,1,0
        for i in range(1, len(s)):
            if s[i] == '0' and (s[i-1]>'2' or s[i-1]=='0'):# 0种
                c = 0
            elif 0<int(s[i-1:i+1])<10 or int(s[i-1:i+1])>26: # 只能单个
               c = b
            elif s[i]=='0' and '0'<s[i-1]<'3':
                c = a    # 只能拼
            else:  # 拼和不拼都可
                c = a+b
            a,b = b,c
        return c


"""
# lc 322 ###零钱兑换(背包问题)
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
你可以认为每种硬币的数量是无限的。
即这一类问题类似砝码问题，给一个target和一个array，array中可以挑出几个数字之和为target，问最少可以选哪几个数字，组成不了则返回-1.

此类问题总结如下：
## 0-1背包问题：每个物品只有一个  有限容量(target)的背包 能装的最大价值(如果是数量最少，可以看做价值全为1，就转为最少价值)是多少。  
这时候用两维dp表  dp[i][j]表示前i个物品在容量为j的情况下能装的最大价值
初始条件即为dp[0][:] = 0， dp[:][0]= 0 即前0个物品最大价值为0，容量为0的时候也是为0。
状态转移方程为dp[i][j] = max(dp[i−1][j], dp[i−1][j−w[i]]+v[i])
1.不装入第i件物品，即dp[i−1][j]
2.装入第i件物品（前提是能装下），即dp[i−1][j−w[i]] + v[i]

## 完全背包问题：每种物品有无限个 有限容量(target)的背包 能装的最大价值是多少  本题即此类型

## 多重背包问题：每种物品超过一个且有限 有限容量(target)的背包 能装的最大价值是多少
 # 此种问题则是将其转化成0-1背包问题。因为每种物体有限个，假设物体[a, b, c]，每种都有两个， 则将其看做[a,a,b,b,c,c]然后就是一个
 0-1背包问题了


"""
class Solution_3:
    def coinChange(self, coins, amount: int) -> int:
        # 动态规划的迭代版本
        dp = [amount+1] * (amount+1)
        # base case
        dp[0] = 0
        for i in range(1, amount+1):  # 枚举总金额
            for coin in coins:  # 枚举每一枚硬币数
                if i-coin < 0:
                    continue
                dp[i] = min(dp[i], dp[i-coin]+1)  # 每一枚硬币都试一下  挑最小的
        return dp[amount] if dp[amount] != amount+1 else -1

"""
# lc 139 单词拆分:
给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词
dp[i]表示s[:i+1]的字符串能否被拆分成单词  base case dp[0]=1 每次判断dp[i]时，要用j遍历0~i-1之间的字符判断j~i-1之间的串能否组成单词然后和dp[j]相与
即转移方程 dp[i] = dp[j] & nums[j+1:i] in dict
"""
class Solution_4:
    def wordBreak(self, s: str, wordDict) -> bool:
        dp = [0] * (1 + len(s))  # dp[0] 作为base case  从dp[1]开始nums[0]的状态
        dp[0] = 1
        word_max = 0
        for word in wordDict:
            word_max = max(word_max, len(word))
        for i in range(1, len(s)+1):
            for j in range(i-1, -1, -1):
                if j-i+1 > word_max:
                    break
                if dp[j] and (s[j:i] in set(wordDict)):
                    dp[i] = 1
                    break
        return True if dp[len(s)] else False

# 正则表达式


# 最大连续上升子序列
def lengthOfLIS( nums) -> int:
    # dp法 时间复杂度是On^2
    # dp[i] 表示nums[0:i]字符子串的最长递增子序列长度
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)  # j遍历0~i-1的字符串的时候，在满足条件的情况下 不断选取dp[i] 和 dp[j]+1 之间的较大值
    return max(dp)


# 连续子数组的最大和
# 即给一个数组，求所有子数组中和最大的 子数组的和
def maxSubArray(nums):
    # dp  定义dp[i]为以nums[i]结尾的子数组 的最大和，然后只要返回dp数组的最大值。每次计算dp[i]时，看dp[i-1]是否为正，即是否对
    # dp[i]起增大作用。转移方程 dp[i] = nums[i]+max(dp[i-1], 0)
    # 因为只用到当前和之前的nums，因此为节省空间，nums数组直接当做dp数组
    for i in range(1, len(nums)):
        nums[i] = nums[i] + max(nums[i-1], 0)
    return max(nums)


# 打家劫舍
# 完美平方
# 最小划分
# 跳跃游戏
# lc 54 螺旋矩阵  顺时针旋转矩阵  a = list(zip(*a[::-1]))   逆时针旋转 a = list(zip(*a))[::-1]

# 记录一下二维矩阵中的路径的相关问题
# 一般有以下一些问题 是否能从a点到达b点， a点到b点的路径有哪些，a点到b点最短路径（或者路径和最小即路径带权）

def find(x1, y1, x2, y2, map):
    # 起始点和终点坐标分别为(x1,y1) (x2,y2), map上0是障碍物  问是否能从起点到终点
    n, m = len(map), len(map[0])
    memo = [[0] * m for _ in range(n)]
    def move(x, y, memo):
        if x==x2 and y==y2:
            return True
        tmp = False
        if 0 <= x < n and 0<=y<m and memo[x][y] == 0 and Map[x][y] != 0:
            memo[x][y] = 1
            tmp = move(x+1, y, memo) or move(x-1, y, memo) or move(x, y+1, memo) or move(x, y-1, memo)
            if not tmp:
                memo[x][y] = 0
        return tmp
    res = move(x1, y1, memo)
    return res

