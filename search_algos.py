# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

from heapq import heapify, heappop, heappush
import os
import pickle
import sys


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        node = None
        try:
            node = heappop(self.queue)

        except Exception as ie:
            print
            "The heap is empty!"
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        return node
        # TODO: finish this function!

    def remove(self, node_id):

        for ind, el in enumerate(self.queue):
            if el[1] == node_id:
                self.queue.pop(ind)
                break
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        heappush(self.queue, node)
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        # raise NotImplementedError

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def get_index(self, node_id):
        for ind, el in enumerate(self.queue):
            print
            el[1], node_id
            if el[1] == node_id:
                return ind
        return -1

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    pq = PriorityQueue()
    explored = {}
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    path = []
    if start == goal:
        return []
    qu = []
    explored[start] = 1
    parent = {}
    qu.append(start)
    path.append(goal)
    flag = 1
    try:
        while len(qu) > 0 and flag == 1:
            src = qu.pop(0)
            for node in graph[src]:
                if node != goal and node not in explored:
                    qu.append(node)
                    parent[node] = src
                    explored[node] = 1
                if node == goal:
                    flag = 0
                    parent[node] = src
                    explored[node] = 1
                    break

    except Exception as e:
        print
        "Returned all the nodes, heap is empty."

    key = goal
    while goal in parent.keys():
        path.insert(0, parent[goal])
        goal = parent[goal]
    print
    path
    return path
    # TODO: finish this function!
    # raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    qu = PriorityQueue()
    explored = {}
    frontier = []
    parent_weight = {}
    path = []
    if start == goal:
        return []
    parent_weight[start] = 0
    parent = {}
    qu.append((0, start))
    path.append(goal)
    flag = 1
    try:
        while flag == 1:
            src_node = qu.pop()
            explored[src_node[1]] = 1
            frontier = [x[1] for x in qu.queue]
            parent_weight[src_node[1]] = src_node[0]
            src = src_node[1]
            if src == goal:
                flag = 0
                break
            cnt = 0
            for node in graph[src]:
                if node not in explored and node not in frontier:
                    qu.append((graph[src][node]["weight"] + parent_weight[src], node))
                    frontier.append(node)
                    parent[node] = src
                elif node in frontier and qu.queue[qu.get_index(node)][0] > graph[src][node]["weight"] + parent_weight[
                    src]:
                    qu.remove(node)
                    parent[node] = src
                    qu.append((graph[src][node]["weight"] + parent_weight[src], node))
                cnt += 1

    except Exception as e:
        print
        e

    key = goal
    while goal in parent.keys():
        path.insert(0, parent[goal])
        goal = parent[goal]
    return path


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def custom_dist_heuristic(graph, v, goals):
    dist1 = ((float(graph.node[v]["pos"][0] - graph.node[goals[0]]["pos"][0]) ** 2) + (
                graph.node[v]["pos"][1] - graph.node[goals[1]]["pos"][1]) ** 2) ** 0.5
    dist2 = ((float(graph.node[v]["pos"][0] - graph.node[goals[1]]["pos"][0]) ** 2) + (
                graph.node[v]["pos"][1] - graph.node[goals[1]]["pos"][1]) ** 2) ** 0.5
    return min(dist1, dist2) * 1.1


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """
    return ((float(graph.node[v]["pos"][0] - graph.node[goal]["pos"][0]) ** 2) + (
                graph.node[v]["pos"][1] - graph.node[goal]["pos"][1]) ** 2) ** 0.5


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    qu = PriorityQueue()
    explored = {}
    frontier = []
    parent_weight = {}
    path = []
    if start == goal:
        return []
    parent_weight[start] = 0
    parent = {}
    qu.append((0 + heuristic(graph, start, goal), start, 0))
    path.append(goal)
    flag = 1
    try:
        while flag == 1:
            src_node = qu.pop()
            explored[src_node[1]] = 1
            frontier = [x[1] for x in qu.queue]
            parent_weight[src_node[1]] = src_node[2]
            src = src_node[1]
            if src == goal:
                flag = 0
                break
            cnt = 0
            for node in graph[src]:
                h = heuristic(graph, node, goal)
                if node not in explored and node not in frontier:
                    qu.append((graph[src][node]["weight"] + parent_weight[src] + h, node,
                               graph[src][node]["weight"] + parent_weight[src]))
                    frontier.append(node)
                    parent[node] = src
                elif node in frontier and qu.queue[qu.get_index(node)][0] > graph[src][node]["weight"] + parent_weight[
                    src] + h:
                    qu.remove(node)
                    parent[node] = src
                    qu.append((graph[src][node]["weight"] + parent_weight[src] + h, node,
                               graph[src][node]["weight"] + parent_weight[src]))
                cnt += 1

    except Exception as e:
        print
        e

    key = goal
    while goal in parent.keys():
        path.insert(0, parent[goal])
        goal = parent[goal]
    return path


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # maintain 2 queues one for forward search and another for backward search
    fr_qu = PriorityQueue()
    bk_qu = PriorityQueue()
    fr_explored = {}
    fr_frontier = []
    bk_explored = {}
    bk_frontier = []
    fr_parent_weight = {}
    bk_parent_weight = {}
    bk_src = None
    fr_src = None
    path = []
    if start == goal:
        return []
    fr_parent_weight[start] = 0
    bk_parent_weight[goal] = 0
    bk_parent = {goal: None}
    fr_parent = {start: None}
    fr_qu.append((0, start, 0))
    bk_qu.append((0, goal, 0))
    min_dist = sys.maxint
    common_point = None
    flag = 1

    try:
        while flag == 1:

            fr_src_node = fr_qu.pop()
            fr_explored[fr_src_node[1]] = 1
            fr_frontier = [x[1] for x in fr_qu.queue]
            fr_parent_weight[fr_src_node[1]] = fr_src_node[2]
            fr_src = fr_src_node[1]

            if fr_src == goal or bk_src == start:
                break

            # check if the popped node of the forward search exists in the explored list of backward search
            if fr_src in bk_explored:
                flag = 0
                common_point = fr_src
                min_dist = fr_src_node[2] + bk_parent_weight[fr_src]
                bk_search_list = bk_explored.keys() + bk_frontier
                fk_search_list = fr_explored.keys()
                common = set(bk_search_list).intersection(set(fk_search_list))
                # check if there are any common nodes in backward and forward searches, if there are choose the path with minimum cost.
                for x in common:
                    if x in bk_frontier:
                        cost_bk = bk_qu.queue[bk_qu.get_index(x)][2]
                    else:
                        cost_bk = bk_parent_weight[x]
                    if min_dist > fr_parent_weight[x] + cost_bk:
                        min_dist = fr_parent_weight[x] + cost_bk
                        common_point = x
                break
            # standard search steps
            for node in graph[fr_src]:
                h_fr = heuristic(graph, node, goal)
                if node not in fr_explored and node not in fr_frontier:
                    fr_qu.append((graph[fr_src][node]["weight"] + fr_parent_weight[fr_src] + h_fr, node,
                                  graph[fr_src][node]["weight"] + fr_parent_weight[fr_src]))
                    fr_frontier.append(node)
                    fr_parent[node] = fr_src
                elif node in fr_frontier and fr_qu.queue[fr_qu.get_index(node)][0] > graph[fr_src][node]["weight"] + \
                        fr_parent_weight[fr_src] + h_fr:
                    fr_qu.remove(node)
                    fr_parent[node] = fr_src
                    fr_qu.append((graph[fr_src][node]["weight"] + fr_parent_weight[fr_src] + h_fr, node,
                                  graph[fr_src][node]["weight"] + fr_parent_weight[fr_src]))

            bk_src_node = bk_qu.pop()
            bk_explored[bk_src_node[1]] = 1
            bk_frontier = [x[1] for x in bk_qu.queue]
            bk_parent_weight[bk_src_node[1]] = bk_src_node[2]
            bk_src = bk_src_node[1]

            # same logic as above
            if bk_src in fr_explored:
                flag = 0
                common_point = bk_src
                min_dist = fr_parent_weight[common_point] + bk_src_node[2]
                bk_search_list = bk_explored.keys()
                fk_search_list = fr_explored.keys() + fr_frontier
                common = set(bk_search_list).intersection(set(fk_search_list))
                for x in common:
                    if x in fr_frontier:
                        cost_fr = fr_qu.queue[fr_qu.get_index(x)][2]
                    else:
                        cost_fr = fr_parent_weight[x]
                    if min_dist > cost_fr + bk_parent_weight[x]:
                        min_dist = cost_fr + bk_parent_weight[x]
                        common_point = x
                break
            for node in graph[bk_src]:
                h_bk = heuristic(graph, start, node)
                if node not in bk_explored and node not in bk_frontier:
                    bk_qu.append((graph[bk_src][node]["weight"] + bk_parent_weight[bk_src] + h_bk, node,
                                  graph[bk_src][node]["weight"] + bk_parent_weight[bk_src]))
                    bk_frontier.append(node)
                    bk_parent[node] = bk_src
                elif node in bk_frontier and bk_qu.queue[bk_qu.get_index(node)][0] > graph[bk_src][node]["weight"] + \
                        bk_parent_weight[bk_src] + h_bk:
                    bk_qu.remove(node)
                    bk_parent[node] = bk_src
                    bk_qu.append((graph[bk_src][node]["weight"] + bk_parent_weight[bk_src] + h_bk, node,
                                  graph[bk_src][node]["weight"] + bk_parent_weight[bk_src]))

    except Exception as e:
        print
        e

    # different cases to handle retrieving the final path, the common node could be an intermediary node or the start node or the end node.
    if flag == 0:
        key = common_point
        path.append(key)
        while key != goal:
            path.append(bk_parent[key])
            key = bk_parent[key]
        key = common_point
        while key != start:
            path.insert(0, fr_parent[key])
            key = fr_parent[key]
        return path
    elif fr_src == goal:
        path.append(goal)
        key = goal
        while fr_parent[key] != None:
            path.insert(0, fr_parent[key])
            key = fr_parent[key]
    elif bk_src == start:
        path.append(goal)
        key = goal
        while bk_parent[key] != None:
            path.append(bk_parent[key])
            key = bk_parent[key]
    return path


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    fr_qu = PriorityQueue()
    bk_qu = PriorityQueue()
    fr_explored = {}
    fr_frontier = []
    bk_explored = {}
    bk_frontier = []
    fr_parent_weight = {}
    bk_parent_weight = {}
    bk_src = None
    fr_src = None
    path = []
    path_a = []
    if start == goal:
        return []
    fr_parent_weight[start] = 0
    bk_parent_weight[goal] = 0
    bk_parent = {goal: None}
    fr_parent = {start: None}
    fr_qu.append((0, start))
    bk_qu.append((0, goal))
    min_dist = sys.maxint
    common_point = None
    flag = 1

    try:
        while flag == 1:
            fr_src_node = fr_qu.pop()
            fr_explored[fr_src_node[1]] = 1
            fr_frontier = [x[1] for x in fr_qu.queue]
            fr_parent_weight[fr_src_node[1]] = fr_src_node[0]
            fr_src = fr_src_node[1]
            if fr_src == goal or bk_src == start:
                break
            if fr_src in bk_explored:
                flag = 0
                common_point = fr_src
                min_dist = fr_src_node[0] + bk_parent_weight[fr_src]
                bk_search_list = bk_explored.keys() + bk_frontier
                fk_search_list = fr_explored.keys()
                common = set(bk_search_list).intersection(set(fk_search_list))
                for x in common:
                    if x in bk_frontier:
                        cost_bk = bk_qu.queue[bk_qu.get_index(x)][0]
                    else:
                        cost_bk = bk_parent_weight[x]
                    if min_dist > fr_parent_weight[x] + cost_bk:
                        min_dist = fr_parent_weight[x] + cost_bk
                        common_point = x
                break
            for node in graph[fr_src]:
                if node not in fr_explored and node not in fr_frontier:
                    fr_qu.append((graph[fr_src][node]["weight"] + fr_parent_weight[fr_src], node))
                    fr_frontier.append(node)
                    fr_parent[node] = fr_src
                elif node in fr_frontier and fr_qu.queue[fr_qu.get_index(node)][0] > graph[fr_src][node]["weight"] + \
                        fr_parent_weight[fr_src]:
                    fr_qu.remove(node)
                    fr_parent[node] = fr_src
                    fr_qu.append((graph[fr_src][node]["weight"] + fr_parent_weight[fr_src], node))
            bk_src_node = bk_qu.pop()
            bk_explored[bk_src_node[1]] = 1
            bk_frontier = [x[1] for x in bk_qu.queue]
            bk_parent_weight[bk_src_node[1]] = bk_src_node[0]
            bk_src = bk_src_node[1]
            if bk_src in fr_explored:
                flag = 0
                common_point = bk_src
                min_dist = fr_parent_weight[common_point] + bk_src_node[0]
                bk_search_list = bk_explored.keys()
                fk_search_list = fr_explored.keys() + fr_frontier
                common = set(bk_search_list).intersection(set(fk_search_list))
                for x in common:
                    if x in fr_frontier:
                        cost_fr = fr_qu.queue[fr_qu.get_index(x)][0]
                    else:
                        cost_fr = fr_parent_weight[x]
                    if min_dist > cost_fr + bk_parent_weight[x]:
                        min_dist = cost_fr + bk_parent_weight[x]
                        common_point = x
                break
            for node in graph[bk_src]:
                if node not in bk_explored and node not in bk_frontier:
                    bk_qu.append((graph[bk_src][node]["weight"] + bk_parent_weight[bk_src], node))
                    bk_frontier.append(node)
                    bk_parent[node] = bk_src
                elif node in bk_frontier and bk_qu.queue[bk_qu.get_index(node)][0] > graph[bk_src][node]["weight"] + \
                        bk_parent_weight[bk_src]:
                    bk_qu.remove(node)
                    bk_parent[node] = bk_src
                    bk_qu.append((graph[bk_src][node]["weight"] + bk_parent_weight[bk_src], node))

    except Exception as e:
        print
        e
    if flag == 0:
        key = common_point
        path.append(key)
        while key != goal:
            path.append(bk_parent[key])
            key = bk_parent[key]
        key = common_point
        while key != start:
            path.insert(0, fr_parent[key])
            key = fr_parent[key]
        return path
    elif fr_src == goal:
        path.append(goal)
        key = goal
        while fr_parent[key] != None:
            path.insert(0, fr_parent[key])
            key = fr_parent[key]
    elif bk_src == start:
        path.append(goal)
        key = goal
        while bk_parent[key] != None:
            path.append(bk_parent[key])
            key = bk_parent[key]
    return path


def gen_path(parent_keys, source, goal):
    # retrieve the entire path
    path = []
    path.append(goal)
    key = goal
    # the parent of the source node will always be None
    while parent_keys[key] != None:
        path.insert(0, parent_keys[key])
        key = parent_keys[key]
    return path


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data