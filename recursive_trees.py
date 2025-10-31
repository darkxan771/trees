import numpy as np
import numpy.random as rand
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
sns.set_theme()

from scipy.special import factorial
from .rooted_trees import RootedTree
from .routines import standardisation, code_to_permutation, permutation_to_code
from .random_trees import PlancherelRecursiveTree, UniformRecursiveTree




class RecursiveTree:
    """
    A class for the encoding of a weighted recursive tree. 

    A recursive tree with n nodes is encoded as an array with several
    length n rows.
    
    - each column i corresponds to the node i in [0, n-1]. The root 
      has label i=0.
    - T.parent[i] is the label of the parent of the node i (with,
      by convention, T.parent[0] = -1).
    - T.weight[i] contains a positive integer, the weight of the node i. 
      This is used in particular when grafting new nodes randomly to T.
    - T.size[i] is the size of the subtree based at i. In particular, 
      T.size[0] = n.
    - T.depth[i] is the depth of the node i (distance to the root). In
      particular, T.depth[0] = 0.
    - T.children[i] contains the list of the children of the node i.
    
    """
    def __init__(self, max_size=10**6):
        self.parent = - np.ones(max_size, dtype=int)
        self.weight = np.zeros(max_size, dtype=int)
        self.weight[0] = 1
        self.size =  np.zeros(max_size, dtype=int)
        self.size[0] = 1
        self.depth = - np.ones(max_size, dtype=int)
        self.depth[0] = 0
        self.children = np.empty(max_size, dtype=object)
        self.children[0] = []
        self.limit = max_size

    def __repr__(self):
        return "Recursive tree with size " + str(self.size[0])  
    
    def __hash__(self):
        return hash(self.to_code())

    def __eq__(self, other):
        return (isinstance(other, RecursiveTree) and (self.to_code() == other.to_code()))
    
    def first_columns(self, d=None):
        """
        Returns the d first columns of the array encoding the recursive tree.
        """
        if d == None:
            D = self.size[0]
        else:
            D = d
        matrix = np.array([self.parent[:D], self.weight[:D], self.size[:D],
                           self.depth[:D], self.children[:D]])
        row_names = ["parent", "weight", "size", "depth", "children"]
        column_names = [str(i) for i in range(D)]
        return pd.DataFrame(matrix, columns=column_names, index=row_names)



    ###############
    # Conversions #
    ###############

    def to_code(self):
        """
        Returns the code of the recursive tree (list of the nodes of attachment).
        """
        return tuple(self.parent[1:].tolist())

    def to_permutation(self):
        """
        Returns the permutation of the recursive tree, obtained via one of the
        bijection from recursive trees with size n to permutations with size n-1.
        """
        return tuple((code_to_permutation(self.parent[1:])).tolist())

    def to_networkx(self):
        """
        Encodes the tree in a NetworkX labelled digraph.
        """
        T = nx.DiGraph({i:self.children[i] for i in range(self.size[0])})
        for i in range(self.size[0]):
            T.nodes[i]["label"] = i
            T.nodes[i]["depth"] = self.depth[i]
            T.nodes[i]["weight"] = self.weight[i]
        return T

    def to_rooted_tree(self):
        """
        Forgets the labels and weights and returns the rooted tree structure.
        """
        return RootedTree([self.subtree(c).to_rooted_tree() for c in self.children[0]])



    ###################################
    # Extract statistical information #
    ###################################

    def number_of_edges(self):
        """
        Returns the number of edges of the tree.
        """
        return self.size[0] - 1

    def number_of_vertices(self):
        """
        Returns the size of the tree (number of vertices).
        """
        return self.size[0]

    def height(self):
        """
        Returns the height of the tree (maximal depth of a node).
        """
        return np.max(self.depth)

    def profile(self):
        """
        Returns the profile of the tree (number of nodes on each level).
        """    
        return np.array([np.count_nonzero(self.depth == d) 
                         for d in range(self.height() + 1)])

    def degrees(self):
        """
        Returns the degrees of the nodes of the tree.
        """
        return np.vectorize(lambda L : len(L))(self.children[:self.size[0]])    

    def path_to_root(self, k):
        """
        Returns the unique path from the root to k.
        """
        res = np.zeros(self.depth[k] + 1, dtype=int)
        res[-1] = k
        for i in range(self.depth[k]):
            res[-i-2] = self.parent[res[-i-1]]
        return res

    def subtree_indices(self, k):
        """
        Returns the set of indices in the subtree based at k.
        """
        if self.children[k] == []:
            return [k]
        else:
            L = sum([self.subtree_indices(int(l)) for l in self.children[k]], [k])
            L.sort()
            return L

    def row_positions(self):
        """
        Returns an array with the row positions of the nodes.
        """
        res = np.zeros(self.size[0], dtype=int)
        count = np.zeros(self.height(), dtype=int)
        h = self.height()
        level = [0]
        next_level = self.children[0]
        for d in range(h):
            for x in next_level:
                res[x] = count[d]
                count[d] += 1
            level = next_level
            next_level = sum([self.children[x] for x in level], [])    
        return res

    def distribution(self, statistic="degree", with_weights=False):
        """
        Computes the distribution of a statistic of the nodes of the tree.

        Available statistics are: 
        - the degree of a node (statistic="degree"). 
        - the depth of a node (statistic="depth"). 
        - the size of the subtree based at a node (statistic="size").

        According to the value of the parameter with_weights, the 
        distribution can be computed with respect to the uniform law
        on nodes, or with respect to the law proportional to the weights.
        """
        n = self.size[0]
        if with_weights:
            w = self.weight[:n]/np.sum(self.weight[:n])
        else:
            w = np.ones(n)/n
        if statistic == "degree":
            L_res = max(self.degrees()) + 1
            lambda_res = (lambda i : len(self.children[i]))
        if statistic == "depth":
            L_res = self.height() + 1
            lambda_res = (lambda i : self.depth[i])
        if statistic == "size":
            L_res = max(self.size) + 1
            lambda_res = (lambda i : self.size[i])    
        res = np.zeros(L_res, dtype=float)
        for i in range(n):
            res[lambda_res(i)] += w[i]
        return res

    def mean(self, statistic="degree", with_weights=False):
        """
        Computes the mean of a statistic of the nodes of the tree.
        """
        dist = self.distribution(statistic, with_weights)
        return np.sum(dist * np.arange(dist.size))

    def var(self, statistic="degree", with_weights=False):        
        """
        Computes the variance of a statistic of the nodes of the tree.
        """
        dist = self.distribution(statistic, with_weights)
        EX2 = np.sum(dist * np.arange(dist.size)**2)
        EX = np.sum(dist * np.arange(dist.size))
        return EX2 - EX**2



    ##############################
    # Modifications / insertions #
    ##############################

    def map_weights(self, func):
        """
        Maps a function on the set of weights.
        """
        n = self.size[0]
        self.weight[:n] = np.vectorize(func)(self.weight[:n])

    def add_node(self, parent_label, **kwargs):
        """
        Adds a node with label (n = size of the tree) as a child of 
        (i = parent_label). 

        Additionnally:
        - a function map_weights can be applied to the weights 
          before the grafting.
        - the weight of the new node can be specified with the argument
          new_weight.
        """
        n = self.size[0] 
        self.parent[n] = parent_label
        self.depth[n] = self.depth[parent_label] + 1
        self.children[n] = [] 
        self.children[parent_label].append(n)
        for k in self.path_to_root(n):
            self.size[k] += 1
        if "map_weights" in kwargs:
            self.map_weights(kwargs["map_weights"])
        if "new_weight" in kwargs:
            self.weight[n] = kwargs["new_weight"]
        else:
            self.weight[n] = 1

    def KP_insertion(self, i, J):
        """
        Realises the Kuba-Panholzer insertion at node i and with weight J.

        The rules are as follows:
        - if the existing tree has size n,  we add the node with label n 
          over the node with label i in [0, n-1].
        - the new node is given the weight J in [1, w(i)].
        - we raise by 1 all the weights j>=J (except for the new node).
        """
        self.add_node(i, map_weights=(lambda x : x + int(x >= J)), new_weight=J)

    def subtree(self, k, normalise_weights=False):
        """
        Returns the recursive subtree based at k.

        According to the value of the parameter normalise_weights, 
        the weights can be recomputed to be a standardisation of 
        the original set of weights.
        """
        sub = np.array(self.subtree_indices(k))
        T = RecursiveTree(max_size = self.size[k])
        dict_sub = {sub[i]:i for i in range(self.size[k])}
        for i in range(self.size[k]):
            if i>0:
                T.parent[i] = dict_sub[self.parent[sub[i]]]
            else:
                T.parent[i] = -1
            T.children[i] = [dict_sub[int(l)] for l in self.children[sub[i]]]
        T.size = self.size[sub]
        T.depth = self.depth[sub] - self.depth[k]
        T.weight = self.weight[sub] 
        if normalise_weights:
            T.weight = standardisation(T.weight) 
        return T

    def cut(self, k, normalise_weights=False):
        """
        Removes the subtree based at k, and renormalises the labels.

        According to the value of the parameter normalise_weights, 
        the weights can be recomputed to be a standardisation of 
        the original set of weights.
        """
        n = self.size[0]
        sub = self.subtree_indices(k)
        sub.remove(k)
        to_substrack  = len(sub)
        keep = [i for i in range(n) if (not (i in sub))]
        dict_keep = {keep[i]:i for i in range(len(keep))}
        dict_keep[-1] = -1
        T = RecursiveTree(max_size = len(keep))
        T.parent = self.parent[keep]
        T.parent = np.vectorize(lambda i : dict_keep[i])(T.parent)
        T.children = self.children[keep]
        T.children[dict_keep[k]] = []
        for i in range(len(keep)):
            T.children[i] = list(map(lambda i : dict_keep[i], T.children[i]))
        T.size = self.size[keep]
        for i in self.path_to_root(k):
            T.size[dict_keep[i]] -= to_substrack
        T.depth = self.depth[keep]
        T.weight = self.weight[keep]
        if normalise_weights:
            T.weight = standardisation(T.weight) 
        return T



    ##################
    # Random objects #
    ##################

    def random_node(self, with_weights=False):
        """
        Picks a random node of the tree.

        According to the value of the parameter with_weights, the node i 
        can be chosen uniformly, or with probability proportional to the 
        weight w[i].
        """
        n = self.size[0]
        if with_weights:
            return rand.choice(n, p=self.weight[:n]/sum(self.weight[:n]))
        else:
            return rand.randint(0, n)

    def random_subtree(self, with_weights=False, normalise_weights=False):
        """
        Picks a random node and returns the corresponding recursive subtree.

        The parameter with_weights decides how the node is chosen randomly,
        and the parameter normalise_weights, when set to True, replaces the
        weights of the subtree by a standardisation of this set.
        """
        k = self.random_node(with_weights)
        return self.subtree(k, normalise_weights)

    def random_cut(self, with_weights=False, normalise_weights=False):
        """
        Picks a random node and returns the corresponding cut.

        The parameter with_weights decides how the node is chosen randomly,
        and the parameter normalise_weights, when set to True, replaces the
        weights of the subtree by a standardisation of this set.
        """
        k = self.random_node(with_weights)
        return self.cut(k, normalise_weights)

    def random_leaf(self, subtree=None):
        """
        Picks a leaf at random according to the hook algorithm.

        The optional argument allows one to apply the algorithm to
        a subtree.
        """
        if subtree == None:
            I = list(range(self.size[0]))
        else:
            I = subtree
        L = [k for k in self.subtree_indices(rand.choice(I)) if k in I]
        while len(L) > 1:
            r = rand.choice(L[1:])
            L = [k for k in self.subtree_indices(r) if k in I]
        return int(L[0])

    def random_relabelling(self):
        """
        Relabels the nodes of the tree, by choosing uniformly at random
        among all the possible increasing labellings of the underlying
        rooted tree.
        """
        n = self.size[0]
        d = np.zeros(n, dtype=int)
        dinv = np.zeros(n, dtype=int)
        I = list(range(n))
        while n>0:
            l = self.random_leaf(subtree = I)
            I.remove(l)
            d[n-1] = l
            dinv[l] = n-1
            n -= 1
        n = self.size[0]    
        self.parent[1:n] = dinv[self.parent[d[1:n]]]
        self.weight[:n] = self.weight[d]
        self.size[:n] = self.size[d]
        self.depth[:n] = self.depth[d]
        new_children = np.empty(n, dtype=object)
        for k in range(n):
            new_children[k] = dinv[self.children[d[k]]].tolist()
        self.children[:n] = new_children
        


    #################
    # Visualisation #
    #################

    def plot_distribution(self, statistic="degree", with_weights=False, limit=None):
        """
        Plots the histogram of the distribution of a statistic of the nodes.
        """
        dist = self.distribution(statistic, with_weights)
        fig, ax = plt.subplots()
        if limit==None:
            L = len(dist)
        else:
            L = min(limit, len(dist))
        ax.bar(np.arange(L), dist[:L])
        ax.set_xticks(np.arange(L))
        S = statistic
        if with_weights:
            ax.set_title(f"Weighted distribution of the {S} of a random node")
        else:
            ax.set_title(f"Distribution of the {S} of a random node")
        plt.show()

    def _angles(self):
        n = self.size[0]
        angle_min = np.zeros(n, dtype=float)
        angle_max = np.zeros(n, dtype=float)
        angle_max[0] = 2 * np.pi
        for i in range(1, n):
            p = self.parent[i]
            j = self.children[p].index(i)
            tmin = sum(self.size[k] for k in self.children[p][:j]) / (self.size[p]-1)
            tmax = sum(self.size[k] for k in self.children[p][:j+1]) / (self.size[p]-1)
            if p == 0:
                angle_min[i] = tmin * 2 * np.pi
                angle_max[i] = tmax * 2 * np.pi
            else:
                angle_min[i] = angle_min[p] + (0.1 + 0.8*tmin) * (angle_max[p] - angle_min[p])
                angle_max[i] = angle_min[p] + (0.1 + 0.8*tmax) * (angle_max[p] - angle_min[p])
        return (angle_min + angle_max)/2

    def layout(self, style="centered"):
        """
        Computes a layout for the drawing of the tree.
        """
        n = self.size[0]
        profile = self.profile()
        positions = self.row_positions()
        if style == "centered":
            return {i:np.array([-(profile[self.depth[i]] + 1)/2 + positions[i],
                    self.depth[i]]) for i in range(n)}
        if style == "left-aligned":
            return {i:np.array([positions[i], self.depth[i]]) for i in range(n)}
        if style in ["circular", "log-circular"]:
            if style == "circular":
                radius = self.depth[:n]
            if style == "log-circular":
                radius = np.log(1 + self.depth[:n])   
            angle = self._angles()
            return {i:np.array([radius[i] * np.cos(angle[i]), 
                radius[i] * np.sin(angle[i])]) for i in range(n)} 
        if style == "natural":
            return nx.spring_layout(self.to_networkx(), pos=self.layout("circular"), k=0.1, iterations=300)

    def draw_on_ax(self, ax0, style="centered", labels="simple", with_circles=False,
        **kwargs):
        """
        Draws the tree on a Matplotlib Axes object.
        """
        T = self.to_networkx()
        ax0.set_axis_off()
        n = self.size[0]
        if with_circles:
            for i in range(self.height()+1):
                ax0.add_patch(plt.Circle((0,0), 
                    i, fill=False, edgecolor=(0.8, 0.8, 0.8)))
        if labels == "empty":
            nx.draw_networkx(T, ax=ax0, pos=self.layout(style), 
               with_labels=False, **kwargs)
        if labels == "simple":
            nx.draw_networkx(T, ax=ax0, pos=self.layout(style), **kwargs)
        if labels == "with_weights":
            L = {i:str(i)+":"+str(int(self.weight[i])) for i in range(n)}
            nx.draw_networkx(T, ax=ax0, pos=self.layout(style), labels=L, **kwargs)
        if labels == "double":
            L = {i:str(i)+"|"+str(int(n - self.weight[i])) for i in range(n)}
            nx.draw_networkx(T, ax=ax0, pos=self.layout(style), labels=L, **kwargs)


    def plot(self, style="centered", labels="simple", with_circles=False, **kwargs):
        """
        Plots the recursive tree. 

        Available options:

        - style: "centered", "left-aligned", "natural",
                  "circular", "log-circular".
        - labels: "empty", "simple", "with_weights", "double".
        - with_circles: bool.
        """
        fig, ax0 = plt.subplots(figsize=(8, 8))
        if style in ["circular", "log-circular", "natural"]:
            ax0.set_aspect(1)
        self.draw_on_ax(ax0, style, labels, with_circles, **kwargs)
        plt.show()











class _RecursiveTreesIterator:
    def __init__(self, n):
        self.order = n
        self.current = np.zeros(n - 1, dtype=int)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RecursiveTrees(self.order).from_code(self.current)
        test = np.argwhere(self.current < np.arange(self.order - 1))
        if test.size == 0:
            self.finished = True
        else:
            i = test[-1][0]
            self.current[i] += 1
            self.current[i+1:] = np.zeros(self.order - i - 2, dtype=int)
        return res



class RecursiveTrees:
    """
    A container for recursive trees with a given size n.
    """
    def __init__(self, n):
        self.order = n
        
    def __repr__(self):
        return f"Recursive trees with size {self.order}"

    def __iter__(self):
        return _RecursiveTreesIterator(self.order)

    def __contains__(self, el):
        return (isinstance(el, RecursiveTree) and el.size == self.order)

    def cardinality(self):
        """
        Returns the cardinality of the set of recursive trees with size n,
        which is equal to (n-1)!.
        """
        return factorial(self.order - 1, True)

    def from_permutation(self, p):
        """
        Constructs the unique recursive tree with size n corresponding to
        the permutation p.
        """
        return self.from_code(permutation_to_code(np.array(p)))
               
    def from_code(self, c):
        """
        Construct the unique recursive tree with size n corresponding to
        the code c.
        """
        res = RecursiveTree(max_size = self.order)
        for i in c:
            res.add_node(i)
        return res

    def get_random_element(self, distribution="uniform"):
        """
        Picks a recursive tree at random. Available distributions are:
        "uniform", "plancherel".
        """        
        if distribution=="uniform":
            return UniformRecursiveTree(self.order).get_random_element()
        if distribution=="plancherel":
            return PlancherelRecursiveTree(self.order).get_random_element()











