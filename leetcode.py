# mid
# 字符串的排列
# 长度最小的子数组

# hard
# 滑动窗口最大值
# 最小覆盖子串
# 最小窗口子序列
# 串联所有单词的子串

# 最小覆盖子串和字符串的排列都是滑动窗口方法
# lc 76 hard 最小覆盖子串：
# 给你一个字符串s， 一个字符串t。返回s中涵盖t所有字符的最小子串。 如果s中不存在涵盖t所有字符的子串，则返回空字符串
def minWindow(s, t):
        """
        最小覆盖子串 采用滑动窗口方案  字典target记录t串  下标left，right形成一个窗口，字典windows记录窗口中目标字符的个数
        right一直前进，碰到target中字符，则windows中相应增加一，一旦该字符个数在target中和windows中相等，计数变量valid+1，说明
        此时窗口中包含到一个目标字符。一旦valid==len(t) 则left开始前进删除字符，找最短子串。
        """
        target = {}
        window = {}
        left, right, valid = 0, 0, 0
        min_length = float('inf')
        for i in t:
            target[i] = 1 if i not in target else target[i]+1

        while right < len(s):
            temp_1 = s[right]
            right += 1
            if temp_1 in target:
                window[temp_1] = 1 if temp_1 not in window else window[temp_1]+1
                if window[temp_1] == target[temp_1]:
                    valid += 1

            while valid == len(target):  # 如果window中包含到了目标串
                if right-left < min_length:
                    min_length = right - left  # 保留历史最短长度
                    min_left = left            # 同时保留最短时的左端点
                temp_2 = s[left]
                left += 1
                if temp_2 in target:
                    if window[temp_2] == target[temp_2]:
                        valid -= 1
                    window[temp_2] -= 1
        return s[min_left:min_left+min_length] if min_length != float('inf') else ''


# lc567 mid 字符串的排列
# 给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。换句话说，第一个字符串的排列之一是第二个字符串的 子串
def checkInclusion(s1: str, s2: str) -> bool:
    """
    字符串的排列问题  判断s2是否包含一个s1的排列  同样是滑动窗口方案
    用一个和短串长度相同的定长的窗口去滑动长串，每次都判断一下窗口中的字符是否和短串匹配( valid == len(s1) )
    """
    left = 0
    tar = {}
    for i in s1:
        tar[i] = 1 if i not in tar else tar[i]+1
    while left+len(s1) <= len(s2):
        temp = {}
        valid = 0
        for j in s2[left:left+len(s1)]:
            if j in tar:
                temp[j] = 1 if j not in temp else temp[j]+1
                if tar[j] == temp[j]:
                    valid += 1
            else:
                break
        if valid == len(tar):
            return True
        else:
            left += 1
    return False

# lc 727 最小窗口子序列(vip) hard 是最小覆盖子串的变种，他要求在长串s中，找到最短的子串，使子串的子序列包含目标短串t
# 而覆盖子串则只需找到最短的子串，使之包含目标串的所有字符
def minSubq(s, t):
    """
    二维数组dp[i][j]的值的含义是s[:i+1] 和 t[:j+1] 的最短子序列的起始位置索引
    dp[0][0] = 0 if s[0]==t[0] else -1
    dp[i][0] = i if s[i]==t[0] else -1
    dp[0][j] = -1 因为 j>=1 时，len(s)<len(t)，不存在这样的子串
    dp[i][j] = dp[i-1][j-1] if s[i] == s[j]
             = dp[i-1][j]   if s[i] != s[j]
    知道了最短子串的起始位置，还需要求出其长度
    """
    if not s or not t:
        return None
    m, n = len(s), len(t)
    dp = [[-1] * n for _ in range(m)]
    if s[0] == t[0]:
        dp[0][0] = 0
    for i in range(1, m):
        if s[i] == t[0]:
            dp[i][0] = i
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j-1] if s[i]==t[j] else dp[i-1][j]
    # 下面求最短子串的长度  即遍历长串s下标(最小从n开始)，如果在当前下标i的时候dp值不为-1，即s[:i+1]和t[:n] 有最小子串符合条件，此时下标i就是
    # 最小子串的右端下标，将其减去dp值(起始下标)  即得到子串长度。把s的下标遍历一遍，得到最短长度的子串
    min_len = float('inf')
    for i in range(n, m):
        if dp[i][n-1] != -1:
            if i-dp[i][n-1] + 1 > min_len:
                min_len = i-dp[i][n-1]+1
    res = s[dp[-1][-1]: min_len+1]
    return res


# elephant = list(map(int, input().strip().split()))
# king = list(map(int, input().strip().split()))
elephant = [4,2]
king = [2,5]
gird = [[0] * 9 for _ in range(10)]
# 0<= row <=9 , 0 <= col <= 8
def find(row, col, gird, res):
    if row == king[0]-1 and col == king[1]-1:
        return res
    elif row>9 or row<0 or col >8 or col<0 or gird[row][col]==1:
        return 0

    gird[row][col] = 1
    r1 = find(row+2, col+3, gird, res+1)
    r2 = find(row + 2, col - 3, gird, res + 1)
    r3 = find(row - 2, col + 3, gird, res + 1)
    r4 = find(row - 2, col - 3, gird, res + 1)
    r5 = find(row + 3, col + 2, gird, res + 1)
    r6 = find(row - 3, col + 2, gird, res + 1)
    r7 = find(row + 3, col - 2, gird, res + 1)
    r8 = find(row - 3, col - 2, gird, res + 1)
    gird[row][col] = 0

    return max(r1, r2, r3, r4, r5, r6, r7, r8)

a = find(elephant[0], elephant[1], gird, 0)

if a:
    print(a)
else:
    print(-1)
