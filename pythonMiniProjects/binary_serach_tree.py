class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    # add method runs at O(log2(n)) time
    def add(self, current, value):
        if self.root == None:
            self.root = Node(value)
        else:
            if value < current.value:
                if current.left == None:
                    current.left = Node(value)
                else:
                    self.add(current.left, value)
            else:
                if current.right == None:
                    current.right = Node(value)
                else:
                    self.add(current.right, value)


node1 = Node(5)


