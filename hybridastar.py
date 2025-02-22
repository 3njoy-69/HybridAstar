# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:45:58 2023

@author: HJ
"""

#Thuật toán Hybrid A*
import math
import numpy as np
from dataclasses import dataclass
from itertools import product
from copy import deepcopy
from common import SetQueue, GridMap, tic, toc, limit_angle

# Đọc bản đồ
IMAGE_PATH = 'image1.jpg'  # Đường dẫn hình ảnh gốc
THRESH = 172  # Ngưỡng nhị phân của hình ảnh, phần lớn hơn ngưỡng sẽ được gán giá trị 255, phần nhỏ hơn sẽ được gán giá trị 0
MAP_HIGHT = 70  # Chiều cao map (1)
MAP_WIDTH = 120  # Chiều rộng map (1)

MAP = GridMap(IMAGE_PATH, THRESH, MAP_HIGHT, MAP_WIDTH)  # Đối tượng bản đồ lưới

# Vị trí và hướng lưới hóa
MAP_NORM = 1.0  # Một pixel trên bản đồ đại diện cho bao nhiêu mét (m/1) #! Lỗi: Khi MAP_NORM không bằng 1, nền đồ họa sẽ bị sai lệch
YAW_NORM = math.pi / 6  # Mỗi bao nhiêu radian tính là cùng một góc? (rad/1)

# Cài đặt điểm bắt đầu và điểm kết thúc
START = [5.0, 35.0, -math.pi / 6]  # Điểm bắt đầu (x, y, yaw), trục y hướng xuống là dương, yaw theo chiều kim đồng hồ là dương.
END = [60.0, 65.0, math.pi / 2]  # Điểm kết thúc (x, y, yaw), trục y hướng xuống là dương, yaw theo chiều kim đồng hồ là dương
ERR = 0.5  # Dừng tìm kiếm khi khoảng cách đến điểm kết thúc nhỏ hơn ERR mét.

# Mô hình vehicle
CAR_LENGTH = 4.5  # Chiều dài phương tiện (m)
CAR_WIDTH = 2.0  # Chiều rộng phương tiện (m)
CAR_MAX_STEER = math.radians(30)  # Góc đánh lái tối đa (rad)
CAR_MAX_SPEED = 8  # Tốc độ tối đa (m/s)


# Định nghĩa mô hình chuyển động
def motion_model(s, u, dt):
    """
    >>> u = [v, δ]
    >>> dx/dt = v * cos(θ)
    >>> dy/dt = v * sin(θ)
    >>> dθ/dt = v/L * tan(δ)
    """
    s = deepcopy(s)
    s[0] += u[0] * math.cos(s[2]) * dt
    s[1] += u[0] * math.sin(s[2]) * dt
    s[2] += u[0] / CAR_LENGTH * math.tan(u[1]) * dt
    s[2] = limit_angle(s[2])
    return s


# Node tọa độ
@dataclass(eq=False)
class HybridNode:
    """Node"""

    x: float
    y: float
    yaw: float

    G: float = 0.  # Chi phí G
    cost: float = None  # Chi phí F
    parent: "HybridNode" = None  # Con trỏ parent node

    def __post_init__(self):
        # Lưới hóa tọa độ và hướng
        self.x_idx = round(self.x / MAP_NORM)  # int Lấy phần nguyên xuống, round Làm tròn
        self.y_idx = round(self.y / MAP_NORM)
        self.yaw_idx = round(self.yaw / YAW_NORM)
        if self.cost is None:
            self.cost = self.calculate_heuristic([self.x, self.y], END)

    def __call__(self, u, dt):
        # Tạo node mới -> new_node = node(u, dt)
        x, y, yaw = motion_model([self.x, self.y, self.yaw], u, dt)
        G = self.G + self.calculate_distance([self.x, self.y], [x, y]) + abs(yaw - self.yaw)
        return HybridNode(x, y, yaw, G, parent=self)

    def __eq__(self, other: "HybridNode"):
        # So sánh node -> node in list
        return self.x_idx == other.x_idx and self.y_idx == other.y_idx and self.yaw_idx == other.yaw_idx
        # return self.__hash__() == hash(other)

    def __le__(self, other: "HybridNode"):
        # Chi phí <= So sánh -> min(open_list)
        return self.cost <= other.cost

    def __lt__(self, other: "HybridNode"):
        # Chi phí < So sánh -> min(open_list)
        return self.cost < other.cost

    def __hash__(self) -> int:
        # So sánh hash của node -> node in set
        return hash((self.x_idx, self.y_idx, self.yaw_idx))

    def heuristic(self, TARG=END):
        """Tìm kiếm theo hướng dẫn, tính toán giá trị hướng dẫn H và cập nhật giá trị F"""
        H = self.calculate_heuristic([self.x, self.y], TARG)
        self.cost = self.G + H
        return H

    def is_end(self, err=ERR):
        """Có phải là điểm kết thúc không, nếu giá trị hướng dẫn H nhỏ hơn err"""
        if self.cost - self.G < err:
            return True
        return False

    def in_map(self, map_array=MAP.map_array):
        """Có nằm trong bản đồ không?"""
        return (0 <= self.x < map_array.shape[1]) and (0 <= self.y < map_array.shape[0])  # h*w维, 右边不能取等!!!

    def is_collided(self, map_array=MAP.map_array):
        """Có xảy ra va chạm không?"""
        # Tính toán tọa độ của bốn đỉnh của hộp giới hạn (bounding box) của phương tiện.
        cos_ = math.cos(self.yaw)
        sin_ = math.sin(self.yaw)
        LC = CAR_LENGTH / 2 * cos_
        LS = CAR_LENGTH / 2 * sin_
        WC = CAR_WIDTH / 2 * cos_
        WS = CAR_WIDTH / 2 * sin_
        x1 = self.x + LC + WS
        y1 = self.y - LS + WC
        x2 = self.x + LC - WS
        y2 = self.y - LS - WC
        x3 = self.x - LC + WS
        y3 = self.y + LS + WC
        x4 = self.x - LC - WS
        y4 = self.y + LS - WC
        # Kiểm tra xem các lưới mà hộp giới hạn (bounding box) bao phủ có chứa vật cản và vượt ra ngoài ranh giới không
        for i in range(int(min([x1, x2, x3, x4]) / MAP_NORM), int(max([x1, x2, x3, x4]) / MAP_NORM)):
            for j in range(int(min([y1, y2, y3, y4]) / MAP_NORM), int(max([y1, y2, y3, y4]) / MAP_NORM)):
                if i < 0 or i >= map_array.shape[1]:
                    return True
                if j < 0 or j >= map_array.shape[0]:
                    return True
                if map_array[j, i] == 0:  # Kích thước h*w, y là chỉ số đầu tiên, 0 đại diện cho vật cản
                    return True
        return False

    @staticmethod
    def calculate_distance(P1, P2):
        """Khoảng cách Euclid (Euclidean distance)"""
        return math.hypot(P1[0] - P2[0], P1[1] - P2[1])

    @classmethod
    def calculate_heuristic(cls, P, TARG):
        """Hàm hướng dẫn (Heuristic function)"""
        return cls.calculate_distance(P, TARG)  # Euclidean distance
        # return abs(P[0]-TARG[0]) + abs(P[1]-TARG[1]) # Manhattan distance


""" ---------------------------- Thuật toán Hybrid A* ---------------------------- """


# F = G + H


# Thuật toán Hybrid A*
class HybridAStar:
    """Thuật toán Hybrid A*"""

    def __init__(self, num_speed=3, num_steer=3, move_step=2, dt=0.2):
        """Thuật toán Hybrid A*

        Parameters
        ----------
        num_speed : int
            discrete values of control variable v, num>=1
        num_steer : int
            discrete values of control variable δ, num>=2
        move_step : int
            Số lần tìm kiếm ngược (backward search count)
        dt : float
            Chu kỳ quyết định (Decision cycle)
        """

        # Điểm xuất phát
        self.start = HybridNode(*START)  # Điểm khởi đầu
        self.start.heuristic()  # Cập nhật chi phí F

        # Error Check
        end = HybridNode(*END)
        if not self.start.in_map() or not end.in_map():
            raise ValueError(f"Tọa độ x, y vượt ra ngoài giới hạn bản đồ")
        if self.start.is_collided():
            raise ValueError(f"Tọa độ x hoặc y của điểm xuất phát nằm trên vật cản")
        if end.is_collided():
            raise ValueError(f"Tọa độ x hoặc y của điểm kết thúc nằm trên vật cản")

        # Khởi tạo thuật toán
        self.reset(num_speed, num_steer, move_step, dt)

    def reset(self, num_speed=3, num_steer=3, move_step=2, dt=0.2):
        """Đặt lại thuật toán"""
        self.__reset_flag = False
        assert num_steer > 1, "Số lượng rời rạc của hướng phải lớn hơn 1"
        self.u_all = [
            np.linspace(CAR_MAX_SPEED, 0, num_speed) if num_speed > 1 else np.array([CAR_MAX_SPEED]),
            np.linspace(-CAR_MAX_STEER, CAR_MAX_STEER, num_steer),
        ]
        self.dt = dt
        self.move_step = move_step
        self.close_set = set()  # Lưu trữ vị trí đã đi qua và giá trị G của nó
        self.open_queue = SetQueue()  # Lưu trữ vị trí khả thi xung quanh vị trí hiện tại và giá trị F của chúng
        self.path_list = []  # Lưu trữ lộ trình (dữ liệu trong CloseList không theo thứ tự)

    def search(self):
        """Tìm kiếm lộ trình"""
        return self.__call__()

    def _update_open_list(self, curr: HybridNode):
        """Thêm điểm khả thi vào open_list"""
        for v, delta in product(*self.u_all):
            # Cập nhật node
            next_ = curr
            for _ in range(self.move_step):
                next_ = next_([v, delta], self.dt)  # x, y, yaw, G_cost, parent đã được cập nhật, F_cost chưa được cập nhật

            # Vị trí mới có nằm ngoài bản đồ không?
            if not next_.in_map():
                continue
            # Vị trí mới có va phải vật cản không?
            if next_.is_collided():
                continue
            # Vị trí mới có nằm trong CloseList không?
            if next_ in self.close_set:
                continue

            # Cập nhật chi phí F
            H = next_.heuristic()

            # Thêm/Cập nhật node vào open-list
            self.open_queue.put(next_)

            # Khi khoảng cách còn lại nhỏ, đi chậm lại một chút
            if H < 20:
                self.move_step = 1

    def __call__(self):
        """Tìm kiếm lộ trình A*"""
        assert not self.__reset_flag, "Cần phải reset trước khi call"
        print("Đang tìm kiếm\n")

        # Khởi tạo OpenList
        self.open_queue.put(self.start)

        # Tìm kiếm node theo chiều tiến
        tic()
        while not self.open_queue.empty():
            # Loại bỏ điểm có chi phí F nhỏ nhất trong OpenList
            curr: HybridNode = self.open_queue.get()
            # Cập nhật OpenList
            self._update_open_list(curr)
            # Cập nhật CloseList
            self.close_set.add(curr)
            # Kết thúc vòng lặp
            if curr.is_end():
                break
        print("Tìm kiếm lộ trình hoàn thành\n")
        toc()

        # Các nút được kết hợp thành các đường dẫn
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()

        # Cần thiết lập lại
        self.__reset_flag = True

        return self.path_list


# debug
if __name__ == '__main__':
    p = HybridAStar()()
    MAP.show_path(p)