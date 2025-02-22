# -*- coding: utf-8 -*-
"""
 Created on Fri May 26 2023 16:03:59
 Modified on 2023-5-26 16:03:59

 @auther: HJ https://github.com/zhaohaojie1998
"""
# Các thành phần chung của thuật toán
from typing import Union
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from dataclasses import dataclass, field

Number = Union[int, float]

__all__ = ['tic', 'toc', 'limit_angle', 'GridMap', 'PriorityQueuePro', 'ListQueue', 'SetQueue', 'Node']


# Node tọa độ
@dataclass(eq=False)
class Node:
    """Node"""

    x: int
    y: int
    cost: Number = 0  # Chi phí F
    parent: "Node" = None  # Con trỏ parent node

    def __sub__(self, other) -> int:
        """Tính khoảng cách Manhattan other và node"""
        if isinstance(other, Node):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other Phải là tọa độ hoặc Node")

    def __add__(self, other: Union[tuple, list]) -> "Node":
        """Tạo node mới"""
        x = self.x + other[0]
        y = self.y + other[1]
        cost = self.cost + math.sqrt(other[0] ** 2 + other[1] ** 2)  # Khoảng cách Euclid
        return Node(x, y, cost, self)

    def __eq__(self, other):
        """So sánh tọa độ x, y -> node in list"""
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, (tuple, list)):
            return self.x == other[0] and self.y == other[1]
        return False

    def __le__(self, other: "Node"):
        """So sánh chi phí -> min(open_list)"""
        return self.cost <= other.cost

    def __lt__(self, other: "Node"):
        """So sánh chi phí -> min(open_list)"""
        return self.cost < other.cost

    def __hash__(self) -> int:
        """Làm cho đối tượng có thể băm (hash) được, Có thể đưa vào tập hợp (set) -> node in set"""
        return hash((self.x, self.y))  # tuple có thể băm (hashable)
        # data in set Độ phức tạp thời gian O(1), data có thể băm hash
        # data in list Độ phức tạp thời gian O(n)


# Set Hàng đợi ưu tiên
@dataclass
class SetQueue:
    """Hàng đợi lưu trữ ưu tiên nút dạng set"""

    queue: set[Node] = field(default_factory=set)

    # Queue Tăng cường container hàng đợi
    def __bool__(self):
        """Kiểm tra: while Queue:"""
        return bool(self.queue)

    def __contains__(self, item):
        """pos in Queue"""
        return item in self.queue
        # NOTE: in là so sánh giá trị, chỉ kiểm tra xem hash có trong tập hợp hay không, kiểm tra id có trong tập hợp hay không.

    def __len__(self):
        """Độ dài: len(Queue)"""
        return len(self.queue)

    # PriorityQueue Thao tác
    def get(self):
        """Queue loại bỏ node có chi phí nhỏ nhất"""
        node = min(self.queue)  # O(n)?
        self.queue.remove(node)  # O(1)
        return node

    def put(self, node: Node):
        """Queue thêm/cập nhật node"""
        if node in self.queue:  # O(1)
            qlist = list(self.queue)  # Truy xuất phần tử, set Không thể truy xuất theo chỉ mục, cần chuyển đổi
            idx = qlist.index(node)  # O(n)
            if node.cost < qlist[idx].cost:  # Nếu chi phí của node mới nhỏ hơn, thì thêm node mới
                self.queue.remove(node)  # O(1)
                self.queue.add(node)  # O(1) Node bị loại bỏ và node được thêm vào có cùng hash, nhưng khác cost và parent.
        else:
            self.queue.add(node)  # O(1)

    def empty(self):
        """Queue có rỗng không?"""
        return len(self.queue) == 0


# List Hàng đợi ưu tiên
@dataclass
class ListQueue:
    """Hàng đợi lưu trữ ưu tiên nút dạng list"""

    queue: list[Node] = field(default_factory=list)

    # Tăng cường container hàng đợi (Queue)
    def __bool__(self):
        """Kiểm tra: while Queue:"""
        return bool(self.queue)

    def __contains__(self, item):
        """contains: pos in Queue"""
        return item in self.queue
        # NOTE: in là so sánh giá trị, chỉ kiểm tra xem value có trong danh sách hay không, không kiểm tra id có trong danh sách hay không.

    def __len__(self):
        """Độ dài: len(Queue)"""
        return len(self.queue)

    def __getitem__(self, idx):
        """Truy xuất: Queue[i]"""
        return self.queue[idx]

    # List Thao tác
    def append(self, node: Node):
        """List Thêm node"""
        self.queue.append(node)  # O(1)

    def pop(self, idx=-1):
        """List Lấy ra node"""
        return self.queue.pop(idx)  # O(1) ~ O(n)

    # PriorityQueue Thao tác
    def get(self):
        """Queue Lấy ra node có chi phí nhỏ nhất"""
        idx = self.queue.index(min(self.queue))  # O(n) + O(n)
        return self.queue.pop(idx)  # O(1) ~ O(n)

    def put(self, node: Node):
        """Queue Thêm/Cập nhật node"""
        if node in self.queue:  # O(n)
            idx = self.queue.index(node)  # O(n)
            if node.cost < self.queue[idx].cost:  # Chi phí node mới nhỏ hơn
                self.queue[idx].cost = node.cost  # O(1) Cập nhật chi phí
                self.queue[idx].parent = node.parent  # O(1) Cập nhật parent node
        else:
            self.queue.append(node)  # O(1)

        # NOTE Mặc dù cú pháp try có độ phức tạp thời gian thấp hơn, nhưng nếu ngoại lệ xảy ra thường xuyên, tốc độ thực thi có thể chậm hơn
        # try:
        #     idx = self.queue.index(node)             # O(n)
        #     if node.cost < self.queue[idx].cost:     # Nếu chi phí của nút mới nhỏ hơn
        #         self.queue[idx].cost = node.cost     # O(1) Cập nhật chi phí
        #         self.queue[idx].parent = node.parent # O(1) Cập nhật parent node
        # except ValueError:
        #     self.queue.append(node)                  # O(1)

    def empty(self):
        """Queue Có rỗng không? """
        return len(self.queue) == 0


# Nguyên bản cũng được triển khai bằng list, nhưng get nhanh hơn, còn put chậm hơn
class PriorityQueuePro(PriorityQueue):
    """Hàng đợi lưu trữ nút ưu tiên - Phiên bản gốc"""

    # PriorityQueue Thao tác
    def put(self, item, block=True, timeout=None):
        """Queue Thêm/Cập nhật node"""
        if item in self.queue:  # O(n)
            return  # Sửa đổi dữ liệu sẽ phá vỡ cấu trúc cây nhị phân, nên không lưu nữa.
        else:
            super().put(item, block, timeout)  # O(logn)

    # Queue Tăng cường khả năng của container
    def __bool__(self):
        """Kiểm tra: while Queue:"""
        return bool(self.queue)

    def __contains__(self, item):
        """Chứa: pos in Queue"""
        return item in self.queue
        # NOTE: in là so sánh giá trị, chỉ kiểm tra xem value có trong danh sách hay không, không kiểm tra id có trong danh sách hay không.

    def __len__(self):
        """Độ dài: len(Queue)"""
        return len(self.queue)

    def __getitem__(self, idx):
        """Index: Queue[i]"""
        return self.queue[idx]


# Xử lý ảnh để tạo bản đồ lưới
class GridMap:
    """Trích xuất bản đồ lưới từ hình ảnh"""

    def __init__(
            self,
            img_path: str,
            thresh: int,
            high: int,
            width: int,
    ):
        """Trích xuất bản đồ lưới

        Parameters
        ----------
        img_path : str
            Đường dẫn của ảnh gốc
        thresh : int
            Ngưỡng nhị phân hóa ảnh, các phần có giá trị lớn hơn ngưỡng sẽ được đặt thành 255, phần nhỏ hơn sẽ thành 0
        high : int
            Chiều cao của bản đồ lưới
        width : int
            Chiều rộng của bản đồ lưới
        """
        # Lưu trữ đường dẫn
        self.__map_path = 'map.png'  # Đường dẫn bản đồ lưới
        self.__path_path = 'path.png'  # Đường dẫn kết quả lập kế hoạch đường đi

        # Xử lý ảnh #  NOTE cv2 lưu trữ ảnh theo định dạng HWC
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh gốc H, W, C
        thresh, map_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)  # Nhị phân hóa bản đồ
        map_img = cv2.resize(map_img, (width, high))  # Thiết lập kích thước bản đồ
        cv2.imwrite(self.__map_path, map_img)  # Lưu bản đồ nhị phân

        # Thuộc tính bản đồ lưới
        self.map_array = np.array(map_img)
        """Bản đồ dưới dạng ndarray, kích thước H×W, với 0 đại diện cho chướng ngại vật"""
        self.high = high
        """Chiều cao của bản đồ dưới dạng ndarray"""
        self.width = width
        """Chiều rộng của bản đồ dưới dạng ndarray."""

    def show_path(self, path_list, *, save=False):
        """Vẽ kết quả lập kế hoạch đường đi

        Parameters
        ----------
        path_list : list[Node]
            Danh sách các nút tạo thành đường đi, yêu cầu Node có thuộc tính x, y.
        save : bool, optional
            Có lưu ảnh kết quả hay không.
        """

        if not path_list:
            print("\nTruyền vào danh sách rỗng, không thể vẽ được\n")
            return
        if not hasattr(path_list[0], "x") or not hasattr(path_list[0], "y"):
            print("\nDanh sách các nút đường đi không có thuộc tính tọa độ x hoặc y, không thể vẽ được\n")
            return

        x, y = [], []
        for p in path_list:
            x.append(p.x)
            y.append(p.y)

        fig, ax = plt.subplots()
        map_ = cv2.imread(self.__map_path)
        map_ = cv2.resize(map_, (self.width, self.high))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # R G B
        # img = img[:, :, ::-1] # R G B
        map_ = map_[::-1]  # 画出来的鸡哥是反的, 需要转过来
        ax.imshow(map_, extent=[0, self.width, 0, self.high])  # extent[x_min, x_max, y_min, y_max]
        ax.plot(x, y, c='r', label='path', linewidth=2)
        ax.scatter(x[0], y[0], c='c', marker='o', label='start', s=40, linewidth=2)
        ax.scatter(x[-1], y[-1], c='c', marker='x', label='end', s=40, linewidth=2)
        ax.invert_yaxis()  # đảo ngược trục y
        ax.legend().set_draggable(True)
        plt.show()
        if save:
            plt.savefig(self.__path_path)


# bộ đếm thời gian trong Matlab
def tic():
    '''bắt đầu đếm thời gian'''
    if 'global_tic_time' not in globals():
        global global_tic_time
        global_tic_time = []
    global_tic_time.append(time.time())


def toc(name='', *, CN=True, digit=6):
    '''kết thúc đếm thời gian'''
    if 'global_tic_time' not in globals() or not global_tic_time:  # Chưa thiết lập biến toàn cục hoặc biến toàn cục là []
        print('Chưa được thiết lập tic' if CN else 'tic not set')
        return
    name = name + ' ' if (name and not CN) else name
    if CN:
        print('%sThời gian đã trôi qua %f Giây。\n' % (name, round(time.time() - global_tic_time.pop(), digit)))
    else:
        print('%sElapsed time is %f seconds.\n' % (name, round(time.time() - global_tic_time.pop(), digit)))


# 角度归一化
def limit_angle(x, mode=1):
    """
    mode1 : (-inf, inf) -> (-π, π]
    mode2 : (-inf, inf) -> [0, 2π)
    """
    x = x - x // (2 * math.pi) * 2 * math.pi  # any -> [0, 2π)
    if mode == 1 and x > math.pi:
        return x - 2 * math.pi  # [0, 2π) -> (-π, π]
    return x






