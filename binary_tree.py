class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def isBalanced(root:TreeNode)->bool:
    if not root:
        return True

    def height(root):
        if root is None:
            return 0
        l_h = height(root.left)
        r_h = height(root.right)
        return max(l_h, r_h) + 1

    if abs(height(root.left)-height(root.right)) > 1:
        return False
    else:
        return isBalanced(root.left) and isBalanced(root.right)


def minDepth(root: TreeNode) -> int:
    """
    采用BFS方案，其实也是层次遍历，一旦搜到叶子节点(left and right 都为None)，返回的深度则为最小深度
    :param root:
    :return:
    """
    if not root:
        return 0
    depth = 1
    que = [root]
    while que:
        size = len(que)
        for i in range(size):
            tempt = que.pop(0)
            if tempt.left is None and tempt.right is None:
                return depth
            que.append(tempt.left)
            que.append(tempt.right)
        depth += 1

    return depth


def openLock(deadends, target):
    """
    打开转盘锁，看做一个层序遍历多叉树，每走一层则步数加一  采用BFS方案  遍历所有可能的方式  遇到deadends则跳过，
    碰到target则直接返回深度(步数）
    同时记录已被访问过的记录
    :param deadends:
    :param target:
    :return:
    """
    # 首先定义一个迭代器，来迭代出节点的8个邻居
    def neibor(node:str):
        for i in range(4):
            d = int(node[i])
            for j in [-1,1]:
                n_d = str((d+j) % 10)
                yield node[:i] + n_d + node[i+1:]

    # deadends 列表转为字典
    dic_dead  = {}
    for dead in deadends:
        dic_dead[dead] = 1
    depth = 0
    que = ['0000']
    visited = {}
    while que:
        for i in range(len(que)):
            temp = que.pop(0)
            if temp in dic_dead:  # 如果在死亡名单上，则跳过
                continue
            if temp == target:   # 如果到了目标，则返回深度
                return depth

            # 下面记录没在visited上的节点
            for node in neibor(temp):
                if node not in visited:
                    que.append(node)
                    visited[node] = 1

        depth += 1   # 走完此层，说明此步到不了目标，步数加一

    # 所有队列都遍历完了还没找到则说明不存在解锁方式
    return -1


"""
滑动窗口类题目
"""
def minWindow( s: str, t: str) -> str:
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
            assert left < len(s), 'left={}, valid={}'.format(left, valid)
            temp_2 = s[left]
            left += 1
            if temp_2 in target:
                # 必须得先确保当前的字符在窗口里的数量是否和目标串里的相等，再做valid-1操作，
                # 因为后面是要在window中对这个元素减一的，所以相等了的话肯定不会再和目标串匹配了
                if window[temp_2] == target[temp_2]:
                    valid -= 1
                window[temp_2] -= 1

    return s[min_left:min_left+min_length] if min_length != float('inf') else ''


def checkInclusion( s1: str, s2: str) -> bool:
    """
    字符串的排列问题  判断s2是否包含一个s1的排列  同样是滑动窗口方案
    """
    """version 1"""
    left = 0
    tar = {}
    valid = 0
    for i in s1:
        tar[i] = 1 if i not in tar else tar[i]+1
    while left+len(s1) <= len(s2):
        for j in s2[left:left+len(s1)]:
            if j in tar:
                tar[j] -= 1
                if tar[j] == 0:
                    valid += 1
            else:
                break
        left += 1
    return valid == len(s1)


def findAnagrams( s: str, p: str):
    """
    LeetCode438：找到字符串中所有字母异位次词。
    找到s中 含有p的排列的所有子串，返回子串的首位置  滑动窗口方案

    """
    target = {}
    window = {}
    result = []
    left, right, valid = 0, 0, 0
    for i in p:
        target[i] = 1 if i not in target else target[i]+1
    while right < len(s):
        temp_1 = s[right]
        right += 1
        if temp_1 in target:
            window[temp_1] = 1 if temp_1 not in window else window[temp_1]+1
            if window[temp_1] == target[temp_1]:
                valid += 1
        while right - left == len(p):   # 窗口长度等于p的长度了  开始增加left以缩小窗口
            if valid == len(target):
                result.append(left)
            temp_2 = s[left]
            left += 1
            if temp_2 in target:
                if window[temp_2] == target[temp_2]:
                    valid -= 1
                window[temp_2] -= 1
    return result


def lengthOfLongestSubstring(s):
    left, right, max_len = 0, 0, 0
    window = {}
    while right < len(s):
        temp = s[right]
        if temp not in window:
            window[temp] = 1
        else:
            max_len = max(max_len, right-left)
            while s[left] != temp:
                del window[left]
                left += 1
            del window[left]
            left += 1
        right += 1
    return max_len


def buildTree(self, inorder, postorder) -> TreeNode:
    """
    从中序序列和后序序列中恢复二叉树
    中序做分割，所以出分割点（索引）；前（后）序出节点
    """

    def recover(inorder, start_in, end_in, postorder, start_po, end_po):
        if start_in > end_in:
            return
        root_val = postorder[end_po]
        for i in range(len(inorder)):
            if inorder[i] == root_val:
                index = i
                break
        root = TreeNode(root_val)
        left_len = index - start_in - 1
        root.left = recover(inorder, start_in, index-1, postorder, start_po, start_po+left_len)
        root.right = recover(inorder, index+1, end_in, postorder, start_po+left_len+1, end_po-1)
        return root

    return recover(inorder, 0, len(inorder)-1, postorder, 0, len(postorder)-1)


def swapPairs(self, head: ListNode) -> ListNode:
    root = ListNode(None)
    root.next = head
    p = root
    while p.next and p.next.next:
        p2 = p.next.next
        tmpt = p2.next
        p2.next = p.next
        p.next.next = tmpt
        p.next = p2
        p = p.next.next
    return root.next


if __name__ == '__main__':
    a = [1, 2, 3, 4]
    root = TreeNode(None)
    p = root
    for i in a:
        p.next = TreeNode(i)
        p = p.next
    m = root.next
    # print(m.val)
    # root.next = TreeNode(10)
    # print(m.val)
    q = root
    while q:
        print(q.val)
        q = q.next





