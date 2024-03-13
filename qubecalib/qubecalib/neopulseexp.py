from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, MutableSequence, Optional

import numpy as np
from numpy.typing import NDArray

from .tree import CostedTree, Tree


class SequenceTree:
    def __init__(self) -> None:
        self._tree = CostedTree()
        self._active_node = 0
        self._latest_node = 0
        self._nodes_items: Dict[int, Item] = {}

    def append(self, item: Item) -> int:
        self._latest_node += 1
        self._tree.adopt(
            self._active_node,
            self._latest_node,
            cost=-1,
        )
        self._nodes_items[self._latest_node] = item
        self._active_node = self._latest_node
        return self._latest_node

    def branch(self, branch: Branch) -> Branch:
        # 枝分かれの開始点を退避する
        branched_node = self._active_node
        # blanch を木構造に追加する
        self.append(branch)
        # 枝分かれ終了後の復帰点を退避する
        branch._next_node = self._active_node
        # 枝分かれの開始点を木構造に教える
        self._active_node = branched_node
        # 枝分かれの開始点に Dummy を追加する
        branch_root = self.append(Dummy())
        # branch 以下の最長経路を計算するための開始点を branch に教える
        branch._root_node = branch_root
        return branch

    def finalize(self) -> None:
        # 深い branch から順に slot を配置する
        for _ in [
            self._nodes_items[_]
            for _ in self.breadth_first_search()[1:]
            if isinstance(self._nodes_items[_], Branch)
        ][::-1]:
            _.place(self)

        # 最終的な slot 配置を確定する（SubSequenceを必ず Toplevel に置くなら必要ないかも？）
        # 各アイテムのコストを更新する
        for node, item in self._nodes_items.items():
            self._tree._cost[node] = item.duration
        total_costs = self._tree.evaluate()
        for node, slot in self._nodes_items.items():
            end = total_costs[node]
            if end is None:
                raise ValueError("invalid value of total cost")
            duration = slot.duration
            if duration is None:
                raise ValueError(f"invalid duration of slot {slot}")
            slot.begin = end - duration

    def parentof(self, child: int) -> int:
        return self._tree.parentof(child)

    def breadth_first_search(self, start: Optional[int] = None) -> List[int]:
        return self._tree.breadth_first_search(start)


@dataclass
class RunningConfig:
    contexts: Final[MutableSequence] = deque()


__rc__: Final[RunningConfig] = RunningConfig()


class ContextNode:
    def __init__(self) -> None:
        if len(__rc__.contexts):
            __rc__.contexts[-1].append(self)


class Item:
    def __init__(self, duration: Optional[float] = None) -> None:
        self._duration = duration
        self.begin: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin})"
        )


class Padding(Item):
    def __init__(self, duration: Optional[float] = None) -> None:
        Item.__init__(self, duration)


class Branch(Item):
    def __init__(self) -> None:
        super().__init__()
        self.duration = None
        self._next_node: Optional[int] = None
        self._root_node: Optional[int] = None

    def place(self, tree: SequenceTree) -> None:
        # 最大長を計算する
        for _ in tree.breadth_first_search(self._root_node)[1:]:
            tree._tree._cost[_] = tree._nodes_items[_].duration
        max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
        # branch の duration は最大長に揃えると同時に cost も確定する
        self.duration = max_duration
        if self._next_node is None:
            raise ValueError("_next_node is None")
        tree._tree._cost[self._next_node] = self.duration

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin}, next_node={self._next_node}, root_node={self._root_node})"


class Dummy(Item):
    def __init__(self) -> None:
        super().__init__()
        self._duration = 0

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        """Branch object cannot set duration value"""
        raise ValueError("Branch object cannot set duration value")


class Modifier(Item, ContextNode):
    def __init__(self) -> None:
        super().__init__()
        self.duration = 0
        self.begin: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(begin={self.begin})"


class DequeWithContext(deque):
    def __enter__(self) -> DequeWithContext:
        __rc__.contexts.append(self)
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        __rc__.contexts.pop()


class Sequence(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        self._tree = SequenceTree()
        for item in self:
            if isinstance(item, Item):
                slot = item
                self._tree.append(slot)
            elif isinstance(item, SequenceTree):
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                _tree = item
                all_nodes = self._tree._tree._tree.all
                if all_nodes:
                    offset = max(all_nodes)
                else:
                    offset = 0
                root = _tree._tree._tree.root
                for node, items in _tree._tree._tree.items():
                    if node == root:
                        self._tree._tree._tree[self._tree._active_node] += [
                            _ + offset for _ in items
                        ]
                        continue
                    self._tree._tree._tree[node + offset] = [_ + offset for _ in items]
                for node in _tree.breadth_first_search()[1:]:
                    self._tree._nodes_items[node + offset] = _tree._nodes_items[node]
                    self._tree._tree._cost[node + offset] = -1

    def _get_tree(self) -> Tree:
        if self._tree is None:
            raise ValueError("SequenceTree is not prepared.")
        return self._tree._tree._tree


class SubSequenceBranch(Branch):
    pass


class SubSequence(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のローカルツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = tree.branch(SubSequenceBranch())
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたらブランチツリーとマージする
                _tree = item
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._next_node に次のアイテムをぶら下げる
                _branch = _tree._nodes_items[1]
                if isinstance(_branch, Branch):
                    if _branch._next_node is None:
                        raise ValueError("_next_node is None")
                    next_item = _branch._next_node
                else:
                    raise ValueError("1st node is not Branch")
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    # branch だったら _next_node や _root_node も新しい node 名に更新
                    if isinstance(_item, Branch):
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset
                # ローカルツリーのカウンターをアップデート
                tree._latest_node = max(tree._tree._tree.all)  # + offset
                # ローカルツリーの次の追加先を先頭 branch の次に指定
                tree._active_node = next_item + offset


class SeriesBranch(Branch):
    pass


class Series(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # この context 用のローカルツリーを作る
        tree = SequenceTree()
        # 外側の context にローカルツリーを渡す
        __rc__.contexts[-1].append(tree)
        # ローカルツリーのルート直下を branch して branch item を登録する
        branch = tree.branch(SeriesBranch())
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたらローカルツリーとマージする
                _tree = item
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._next_node に次のアイテムをぶら下げる
                _branch = _tree._nodes_items[1]
                if isinstance(_branch, Branch):
                    if _branch._next_node is None:
                        raise ValueError("_next_node is None")
                    next_item = _branch._next_node
                else:
                    raise ValueError("1st node is not Branch")
                # ツリーをマージするためにノード番号を更新する
                offset = max(tree._tree._tree.all)
                # 幅優先探索で全アイテムを舐める
                for child in _tree.breadth_first_search()[1:]:
                    parent = _tree._tree._tree.parentof(child)
                    # ブランチ用ツリーに追加する
                    tree._tree.adopt(parent + offset, child + offset, cost=-1)
                    # ブランチ用ツリーのあいてむに登録する
                    tree._nodes_items[child + offset] = _tree._nodes_items[child]
                    # branch だったら _next_node や _root_node も新しい node 名に更新
                    _item = _tree._nodes_items[child]
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset
                # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # 親ツリーの追加先を先頭 branch の次に指定
                tree._active_node = next_item + offset


class FlushleftBranch(Branch):
    pass


class Flushleft(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = tree.branch(FlushleftBranch())
        # with 内で定義された item を舐める
        if branch._root_node is None:
            raise ValueError("_root_node is None")
        tree._active_node = branch._root_node
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                # ただしパラレルなので注意
                tree.append(item)
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたら親ツリーとマージする
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._root_node に次のアイテムをぶら下げる
                _tree = item
                # ツリーをマージするためにノード番号を更新する
                offset = max(tree._tree._tree.all)
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset
                # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # 親ツリーの追加先を先頭 branch の次に指定
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node


class FlushrightBranch(Branch):
    def place(self, tree: SequenceTree) -> None:
        super().place(tree)
        # # blank が 0 の状態で最大長を計算する
        max_duration = self._duration
        # _root_node にぶら下がっている blank node を取得する
        for _ in tree._tree._tree[self._root_node]:
            # blank にぶら下がっているノードの最大長を取得する
            branch_duration = max([_ for _ in tree._tree.evaluate(_).values()])
            # 右揃えになるよう blank を調整するかつ cost も確定する
            tree._nodes_items[_].duration = max_duration - branch_duration
            tree._tree._cost[_] = tree._nodes_items[_].duration


class Flushright(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = tree.branch(FlushrightBranch())
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                # ただしパラレルなので注意
                tree.append(Padding(0))
                tree.append(item)
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたら親ツリーとマージする
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._root_node に次のアイテムをぶら下げる
                tree.append(
                    Padding(0)
                )  # flushright は特別に branch の前に Padding をつける
                _tree = item  # ローカルの SequenceTree
                offset = max(tree._tree._tree.all)
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset  # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # # 親ツリーの追加先を先頭 branch の次に指定
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node


@dataclass
class SampledSequenceBase:
    target_name: str
    first_blank: float = 0  # second
    sampling_period: float = 2e-9  # second

    def asdict(self) -> Dict:
        return {}


@dataclass
class GenSampledSequence(SampledSequenceBase):
    sub_sequences: List[GenSampledSubSequence] = field(default_factory=list)


@dataclass
class GenSampledSubSequence:
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    post_blank: float  # second
    repeat: int

    def asdict(self) -> Dict:
        return {}


@dataclass
class CapSampledSequence(SampledSequenceBase):
    sub_sequences: List[CapSampledSubSequence] = field(default_factory=list)


@dataclass
class CapSampledSubSequence:
    duration: float  # second
    bost_blank: float  # second

    def asdict(self) -> Dict:
        return {}


def ceil(value: float, unit: float = 1) -> float:
    """valueの値を指定したunitの単位でその要素以上の最も近い数値に丸める

    Args:
        value (float): 対象の値
        unit (float, optional): 丸める単位. Defaults to 1.

    Returns:
        float: 丸めた値
    """
    MAGNIFIER = 1_000_000

    # unit が循環小数の場合に丸めなければならない場合がある
    if value % unit < 1e-9:
        return value

    value, unit = int(value * MAGNIFIER), int(unit * MAGNIFIER)
    if value % unit:
        return int((value // unit + 1) * unit) / MAGNIFIER
    else:
        return int((value // unit) * unit) / MAGNIFIER


class TargetHolder:
    def set_target(self, target: str) -> TargetHolder:
        self.target = target
        return self


class Slot(ContextNode, Item):
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__()
        Item.__init__(self, duration)


class Range(ContextNode, Item, TargetHolder):
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__()
        Item.__init__(self, duration)


class Blank(Slot):
    SAMPLING_PERIOD: Final[float] = 2e-9

    @property
    def sampling_points(self) -> np.ndarray:
        if self.begin is None or self.duration is None:
            raise ValueError(f"{self.__class__.__name__}: position is not calculated")
        return np.arange(
            ceil(self.begin, 2),
            ceil(self.begin + self.duration, 2),
            self.SAMPLING_PERIOD,
        )  # sampling points [ns]

    def func(self, t: float) -> float:
        return 0.0

    def ufunc(self, t: float) -> NDArray:
        return np.frompyfunc(self.func, 1, 1)(t).astype(complex)


class SlotWithIQ(Slot):
    SAMPLING_PERIOD: Final[float] = 2e-9

    def __init__(self, duration: Optional[float] = None) -> None:
        self.amplitude = 1.0
        self.phase = 0.0  # radian
        self.__iq__: Optional[NDArray] = None
        # self.__virtual_z_theta__ = 0

        super().__init__(duration=duration)

    def func(self, t: float) -> complex:
        raise NotImplementedError()
        return 0

    def cmag_func(self, t: float) -> complex:
        return self.func(t) * self.amplitude * np.exp(1j * self.phase)

    def ufunc(self, t: NDArray) -> NDArray:
        return np.frompyfunc(self.cmag_func, 1, 1)(t).astype(complex)

    # def virtual_z(self, theta):
    #     self.__virtual_z_theta__ = theta

    @property
    def iq(self) -> Optional[NDArray]:
        if self.begin is None or self.duration is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        self.__iq__ = self.ufunc(self.sampling_points_zero)  # * np.exp(
        # 1j * self.__virtual_z_theta__
        # )
        return self.__iq__

    @property
    def sampling_points(self) -> NDArray:
        if self.begin is None or self.duration is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        return np.arange(
            ceil(self.begin, 2e-9),
            ceil(self.begin + self.duration, 2e-9),
            self.SAMPLING_PERIOD,
        )  # sampling points [ns]

    @property
    def sampling_points_zero(self) -> NDArray:
        if self.begin is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        return self.sampling_points - self.begin  # sampling points [ns]


class RaisedCosFlatTop(SlotWithIQ, TargetHolder):
    """
    Raised Cosine FlatTopパルス

    Attributes
    ----------
    ampl : float
        全体にかかる振幅
    phase : float
        全体にかかる位相[rad]
    rise_time: float
        立ち上がり・立ち下がり時間[ns]
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        rise_time: Optional[float] = None,
    ):
        # self.ampl = 1.0
        # self.phase = 0.0
        self.rise_time = rise_time

        super().__init__(duration)

    def func(self, t: float) -> complex:
        if self.duration is None or self.rise_time is None:
            raise ValueError("duration or rise_time is None")
        flattop_duration = self.duration - self.rise_time * 2

        t1 = 0
        t2 = t1 + self.rise_time  # 立ち上がり完了時刻
        t3 = t2 + flattop_duration  # 立ち下がり開始時刻
        t4 = t3 + self.rise_time  # 立ち下がり完了時刻

        if (t1 <= t) & (t < t2):  # 立ち上がり時間領域の条件ブール値
            v = (1.0 - np.cos(np.pi * (t - t1) / self.rise_time)) / (
                2.0 + 0.0j
            )  # 立ち上がり時間領域
        elif (t2 <= t) & (t < t3):  # 一定値領域の条件ブール値
            v = 1.0 + 0.0j  # 一定値領域
        elif (t3 <= t) & (t < t4):  # 立ち下がり時間領域の条件ブール値
            v = (1.0 - np.cos(np.pi * (t4 - t) / self.rise_time)) / (
                2.0 + 0.0j
            )  # 立ち下がり時間領域
        else:
            v = 0.0 + 0.0j

        return v  # * self.ampl * np.exp(1j * self.phase)


class Rectangle(SlotWithIQ, TargetHolder):
    def __init__(self, duration: Optional[float] = None):
        super().__init__(duration)

    def func(self, t: float) -> complex:
        return 1 + 0j


class Arbit(SlotWithIQ, TargetHolder):
    """ "サンプリング点を直接与えるためのオブジェクト"""

    def __init__(
        self,
        duration: Optional[float] = None,
        init: Optional[complex] = None,
    ):
        if init is None:
            self.init = 0 + 0j
        else:
            self.init = init
        super().__init__(duration)

    # @observe("duration")
    # def notify_duration_change(self, e):
    #     self.__iq__ = np.zeros(int(self.duration // self.SAMPLING_PERIOD)).astype(
    #         complex
    #     )  # iq data

    def ufunc(self, t: Optional[NDArray] = None) -> NDArray:
        """
        iq データを格納している numpy array への参照を返す

        Parameters
        ----------
        t : numpy.ndarray(float)
            与えると対象の時間列に則した点数にサンプルした iq データを返す
        """
        if self.__iq__ is None:
            raise ValueError("__iq__ is None")
        if t is None:
            return self.__iq__
        else:
            if self.begin is None or self.duration is None:
                raise ValueError("begin or duration is None")
            rslt = np.zeros(t.shape).astype(complex)
            b, e = self.begin, self.begin + self.duration
            iq = self.__iq__
            idx = (ceil(b, 2) <= t + b) & (t + b < ceil(e, 2))
            # 開始点が 31.999968 の様に誤差を含む場合に開始点を含む
            # idx[0] = True if ceil(b, 2) - (t + b)[idx][0] < 1e-4 else False
            # 終点が 41.999968 の様に誤差を含む場合に終点を除外する
            # idx[-1] = False if ceil(e,2) - (t + b)[idx][-1] < 1e-4 else True
            q, m = t[idx].shape[0], iq.shape[0]
            n = int(q // m)
            v = (
                np.stack(
                    n
                    * [
                        iq,
                    ]
                )
                .transpose()
                .reshape(n * m)
            )
            o = v.shape[0]

            if q == o:
                rslt[idx] = v
            elif q < o:
                rslt[idx] = v[: (q - o)]
            else:
                idx[(o - q) :] = False
                rslt[idx] = v

            return rslt

    @property
    def iq_array(self) -> NDArray:
        if self.__iq__ is None:
            raise ValueError("__iq__ is None")
        return self.__iq__


if __name__ == "__main__":
    with Sequence() as sequence:
        Slot()
    print(sequence)
