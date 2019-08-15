import numpy as np
import math
import time

class Factor:
    def __init__(self, un_list, ob_list):
        self.un_list = un_list
        self.ob_list = ob_list
        self.un_mask = Factor.list2Bin(un_list)
        self.ob_mask = Factor.list2Bin(ob_list)
        # print("un_mask = %s, ob_mask = %s" %(bin(un_mask), bin(ob_mask)))
        self.setFactorTable(np.zeros([self.ob_mask + 1, self.un_mask + 1]))
        self.query_ob_value = None
    @staticmethod
    def list2Bin(index_list):
        if (index_list):
            bin = np.sum([1 << i for i in index_list])
        else:
            bin = 0
        return(bin)
    @staticmethod
    def bin2List(bin):
        index_list = []
        i = 0
        while (bin):
            if (bin & 1):
                index_list.append(i)
            bin >>= 1
        return(bin)
    @staticmethod
    def bin2Index(bin):
        i = 0
        while (bin):
            if (bin & 1):
                return(i)
            bin >>= 1
        return(None)
    @staticmethod
    def weigh(mask):
        weight = 0
        while (mask):
            weight += mask & 1
            mask >>= 1
        return(weight)
    @staticmethod
    def shiftByMask(num, mask):
        p_m = 0
        res = 0
        while (mask):
            b_m = mask & 1
            b_n = num &  1
            if (b_m):
                res |= b_n << p_m
                num >>= 1
            mask >>= 1
            p_m += 1
        return(res)
    @staticmethod
    def enumValues(mask):
        return([Factor.shiftByMask(num, mask) for num in range(0, 1 << Factor.weigh(mask))])
    def setFactorTable(self, table):
        self.factor_table = table.reshape([self.ob_mask + 1, self.un_mask +1])
    def update(self, data):
        un_value = data & self.un_mask
        ob_value = data & self.ob_mask
        self.factor_table[ob_value][un_value] += 1
    def get(self, un_value, ob_value):
        # print("ob_value = %s, un_value = %s" %(bin(ob_value), bin(un_value)))
        # print(self.factor_table)
        if (np.sum(self.factor_table[ob_value]) != 0):
            return(self.factor_table[ob_value][un_value] / np.sum(self.factor_table[ob_value]))
        else:
            return(0)
    def set(self, un_value, ob_value, numerator):
        self.factor_table[ob_value][un_value] = numerator
    def evaluate(self, assignment):
        un_value = assignment & self.un_mask
        ob_value = assignment & self.ob_mask
        return(self.get(un_value, ob_value))
    def diff(self, other):
        if (self.query_ob_value != None):
            return(np.sum([abs(self.get(un_value, self.query_ob_value) - other.get(un_value, self.query_ob_value)) for un_value in Factor.enumValues(self.un_mask)]))
    def flattern(self):
        if (self.ob_list):
            flattern_un_list = list(set(self.un_list) | set(self.ob_list))
            flattern_factor = Factor(flattern_un_list, [])
            for un_value in Factor.enumValues(flattern_factor.un_mask):
                flattern_factor.factor_table[0][un_value] = self.evaluate(un_value)
            return(flattern_factor)
        else:
            return(self)
    def sum(self, index_list):
        sum_over_mask = Factor.list2Bin(index_list)
        flattern_factor = self.flattern()
        other_mask = flattern_factor.un_mask ^ sum_over_mask
        sum_un_list = list(set(flattern_factor.un_list) - set(index_list))
        sum_factor = Factor(sum_un_list, [])
        for un_value in Factor.enumValues(other_mask):
            sum_factor.factor_table[0][un_value] = np.sum([flattern_factor.evaluate(un_value | sum_over_value) for sum_over_value in Factor.enumValues(sum_over_mask)], axis=0)
        # print("\nSum:")
        # print("Fattern factor table:")
        # print(flattern_factor.factor_table)
        # print("Sum factor table")
        # print(sum_factor.factor_table)
        return(sum_factor)
    def getElement(self, assignment):
        un_value = assignment & self.un_mask
        ob_value = assignment & self.ob_mask
        return(self.factor_table[ob_value][un_value])
    def getCol(self, bin):
        return(self.factor_table[:, bin])
    def prod(self, other):
        if (other == None):
            return(self)
        prod_un_list = list(set(self.un_list) | set(self.ob_list) | set(other.un_list) | set(other.ob_list))
        prod_factor = Factor(prod_un_list, [])
        for un_value in Factor.enumValues(prod_factor.un_mask):
            prod_factor.factor_table[0][un_value] = np.array([self.evaluate(un_value) * other.evaluate(un_value)])
        # print("\nProd:")
        # print("Self factor table:")
        # print(self.factor_table)
        # print("Other factor table")
        # print(other.factor_table)
        # print("Prod factor table")
        # print(prod_factor.factor_table)
        return(prod_factor)
    def printMask(self):
        print("ob_mask = %s" %(np.binary_repr(self.ob_mask, 12)))
        print("un_mask = %s" %(np.binary_repr(self.un_mask, 12)))
    def printFactorTable(self):
        print("Shape: (%d, %d)" %(self.factor_table.shape))
        print("Factor Table:\n%s" %(self.factor_table))
    def __str__(self):
        factor_str = "\t\t"
        # factor_str += "un_list = %s\n" % self.un_list
        # factor_str += "ob_list = %s\n" % self.ob_list
        un_weight = Factor.weigh(self.un_mask)
        ob_weight = Factor.weigh(self.ob_mask)
        # print(self.factor_table)
        for j in range(0, 1 << un_weight):
            factor_str += "%s\t" % (np.binary_repr(j, un_weight))
        factor_str += "\n"
        if (self.ob_mask):
            for i in range(0, 1 << ob_weight):
                factor_str += "%s\t" % (np.binary_repr(i, ob_weight))
                for j in range(0, 1 << un_weight):
                    ob_value = Factor.shiftByMask(i, self.ob_mask)
                    un_value = Factor.shiftByMask(j, self.un_mask)
                    if (self.query_ob_value == None or ob_value == self.query_ob_value):
                        factor_str += "%f\t" % (self.get(un_value, ob_value))
                factor_str += "\n"
        else:
            factor_str += "-\t"
            for j in range(0, 1 << un_weight):
                ob_value = 0
                un_value = Factor.shiftByMask(j, self.un_mask)
                if (self.query_ob_value == None or ob_value == self.query_ob_value):
                    factor_str += "%f\t" % (self.get(un_value, ob_value))
            factor_str += "\n"
        return(factor_str)

class Node:
    def __init__(self, name, index, pa_index):
        self.name = name
        self.index = index
        self.pa_index = pa_index
        self.factor = Factor([index], pa_index) 
        self.value = None
    def train(self, data):
        self.factor.update(data)
    def getFactor(self, assignment):
        return(self.factor.evaluate(assignment))
    def sample(self, pa_value):
        assignment_1 = pa_value | self.factor.un_mask
        p = self.getFactor(assignment_1)
        self.value = np.random.binomial(1, p) << self.index
        # print("Sample node %d, pa_value = %s, p = %f, sample_val = %s" % (self.index, np.binary_repr(pa_value, 12), p, np.binary_repr(self.value, 12)))
        return(self.value)
    def __str__(self):
        node_str = "Node %d: %s\n" % (self.index, self.name)
        node_str += "Parents: %s\n" % (self.pa_index)
        node_str += "Factor table:\n"
        node_str += self.factor.__str__() + "\n"
        return(node_str)

class Clique:
    def __init__(self, node_index_list, parent, sep_list):
        self.node_index_list = node_index_list
        self.parent = parent
        self.children = []
        if (parent):
            parent.addChild(self)
        self.sep_list = sep_list
        self.message = None
    def __str__(self):
        clique_str = "Clique %s\n" %(self.node_index_list)
        if (self.parent):
            clique_str += "Parent Clique %s\n" %(self.parent.node_index_list)
        else:
            clique_str += "Root clique\n"
        clique_str += "Seperator: %s\n" %(self.sep_list)
        if (self.message):
            clique_str += "message: \n%s" %(self.message)
        clique_str += "\n"
        return(clique_str)
    def addChild(self, child):
        self.children.append(child)
    def printTree(self):
        print(self)
        for child in self.children:
            child.printTree()
class DirectedGM:
    def __init__(self):
        self.node_num = 0
        self.node_map = {}
        self.clque_map = {}
    def addNode(self, node):
        self.node_num += 1
        self.node_map[node.index] = node
    def getNode(self, index):
        return(self.node_map[index])
    def getNodeList(self, index_list):
        return([self.getNode(i) for i in index_list])
    def train(self, dataset_filename, limit = None):
        print("Training with %s..."  % (dataset_filename))
        t0 = time.time()
        dataset_fp = open(dataset_filename, "r")
        i = 0
        for line in dataset_fp:
            data = int(line.replace("\n", ""))
            # print("data = %s" % (bin(data)))
            for node in self.node_map.values():
                node.train(data)
            i += 1
            if (i % 100000 == 0):
                print("Trained with %i data points..." % (i))
            if (limit != None and i >= limit):
                break
        dataset_fp.close()
        t1 = time.time()
        runtime = t1 - t0
        print("Finished training. Runtime = %d s" % (runtime))
    def jointDist(self, assignment):
        joint_dist = 1
        for node in self.node_map.values():
            joint_dist *= node.getFactor(assignment)
        return(joint_dist)
    def compare(self, true_dist_filename):
        true_dist_fp = open(true_dist_filename, "r")
        l1_dist = 0
        for i in range(0, 1 << self.node_num):
            line = true_dist_fp.readline().replace("\n", "")
            word_list = line.split("\t")
            p_true = float(word_list[1])
            p_esti = self.jointDist(i)
            # print(p_esti)
            l1_dist += abs(p_true - p_esti)
        true_dist_fp.close()
        return(l1_dist)
    def eliminateVar(self, ve_order, query_list, observe_map):
        ve_factor = None
        # print(observe_map)
        for i in ve_order:
            if (i in observe_map):
                delta_factor = Factor([i], [])
                delta_factor.set(observe_map[i] << i, 0, 1)
                ve_factor = delta_factor.prod(self.getNode(i).factor).prod(ve_factor).sum([i])
            elif (i in query_list):
                ve_factor = self.getNode(i).factor.prod(ve_factor)
            else:
                ve_factor = self.getNode(i).factor.prod(ve_factor).sum([i])
            print("Factor m_%d\n%s" %(i, ve_factor))
            # print("Node %d, Factor:\n%s\n" % (i, ve_factor))
        return(ve_factor)
    def constructCliqueTree(self, ve_order):
        tree_root = None
        parent = None
        sep_list = []
        for i in ve_order:
            node = self.getNode(i)
            index_list = list(set(node.pa_index) | set([i]) | set(sep_list))
            clique = Clique(index_list, parent, sep_list)
            sep_list = list(set(index_list) - set([i]))
            parent = clique
            self.clque_map[i] = clique
            if (tree_root == None):
                tree_root = clique
        return(tree_root)
    def messagePassing(self, ve_order, query_list, observe_map):
        tree_root = self.constructCliqueTree(ve_order)
        msg_in = None
        for i in ve_order:
            if (i in observe_map):
                delta_factor = Factor([i], [])
                delta_factor.set(observe_map[i] << i, 0, 1)
                self.clque_map[i].message = delta_factor.prod(self.getNode(i).factor).prod(msg_in).sum([i])
            elif (i in query_list):
                self.clque_map[i].message = self.getNode(i).factor.prod(msg_in)
            else:
                self.clque_map[i].message = self.getNode(i).factor.prod(msg_in).sum([i])
            # print("Message m_%d\n%s" %(i, self.clque_map[i].message))
            msg_in = self.clque_map[i].message
        # tree_root.printTree()
        return(self.clque_map[i].message)
    def query(self, query_list, observe_map, method = "BruteForce", true_dist_filename = None, ve_order = []):
        t0 = time.time()

        result = Factor(query_list, observe_map.keys())
        ob_value = np.sum([val << pos for pos, val in observe_map.items()])
        result.query_ob_value = ob_value
        ot_mask = (1 << self.node_num) - 1 - (result.un_mask | result.ob_mask)
        ot_un_mask = ot_mask | result.un_mask
        if (method == "TrueDistribution" and true_dist_filename != None):
            true_dist_fp = open(true_dist_filename, "r")
            true_dist_table = [float(line.replace("\n", "").split("\t")[1]) for line in true_dist_fp]
        elif (method == "VariableElimination" and ve_order != []):
            ve_factor = self.eliminateVar(ve_order, query_list, observe_map)
        elif (method == "MessagePassing" and ve_order != []):
            mp_factor = self.messagePassing(ve_order, query_list, observe_map)
        for un_value in Factor.enumValues(result.un_mask):
            if (method == "BruteForce"):
                numerator = np.sum([self.jointDist(ot_value | un_value | ob_value) for ot_value in Factor.enumValues(ot_mask)])
                denominator = np.sum([self.jointDist(ot_un_value | ob_value) for ot_un_value in Factor.enumValues(ot_un_mask)])
                prob = numerator / denominator
            elif (method == "TrueDistribution"):
                numerator = np.sum([true_dist_table[ot_value | un_value | ob_value] for ot_value in Factor.enumValues(ot_mask)])
                denominator = np.sum([true_dist_table[ot_un_value | ob_value] for ot_un_value in Factor.enumValues(ot_un_mask)])
                prob = numerator / denominator
            elif (method == "VariableElimination"):
                prob = ve_factor.get(un_value, 0)
            elif (method == "MessagePassing"):
                prob = mp_factor.get(un_value, 0)
            result.set(un_value, ob_value, prob)
        
        t1 = time.time()
        runtime = t1 - t0
        print("Query: %s, given: %s, method: %s, runtime: %s s, result:\n %s" %(query_list, observe_map, method, runtime, result))
        return(result)
    def singleSample(self):
        node_stack = []
        sample_val = 0
        for node in self.node_map.values():
            if (node.pa_index == []):
                sample_val |= node.sample(0)
            else:
                node_stack.append(node)
                while (node_stack):
                    cur_node = node_stack.pop()
                    pa_value = 0
                    pa_stack = []
                    for j in cur_node.pa_index:
                        if (self.getNode(j).value != None):
                            pa_value |= self.getNode(j).value
                        else:
                            pa_stack.append(self.getNode(j))
                    if (pa_stack):
                        node_stack += pa_stack
                    else:
                        sample_val |= cur_node.sample(pa_value)
        return(sample_val)
    def sample(self, sample_filename, sample_num = 1):
        sample_fp = open(sample_filename, "w")
        for i in range(0, sample_num):
            sample_fp.write("%d\n" %(self.singleSample()))
        sample_fp.close()
        return(i)
    def __str__(self):
        gm_str = ""
        for node in self.node_map.values():
            gm_str += node.__str__()
        return(gm_str)

if __name__ == "__main__":
    # Problem 4.1(a): Setup the directed graphic model with naive ideas and random parameters
    np.random.seed(2019)
    gm = DirectedGM()
    gm.addNode(Node("IsSummer", 0, []))
    gm.addNode(Node("HasFlu", 1, [0]))
    gm.addNode(Node("HasFoodPoisoning", 2, [0]))
    gm.addNode(Node("HasHayFever", 3, [0]))
    gm.addNode(Node("HasPneumonia", 4, [0]))
    gm.addNode(Node("HasRespiratoryProblems", 5, [1, 3, 4]))
    gm.addNode(Node("HasGastricProblems", 6, [2]))
    gm.addNode(Node("HasRash", 7, [2, 3]))
    gm.addNode(Node("Coughs", 8, [1, 3, 4]))
    gm.addNode(Node("IsFatigued", 9, [1, 2, 3, 4]))
    gm.addNode(Node("Vomits", 10, [1, 3, 4]))
    gm.addNode(Node("HasFever", 11, [1, 2, 3, 4]))
    print("Problem 4.1(a): Setup the directed graphic model with naive ideas and random parameters")
    print(gm)

    # Problem 4.1(b): Estimate parameters, train the graphical model with the given dataset
    gm.train("dataset.dat")
    print("Problem 4.1(b): Estimate parameters, train the graphical model with the given dataset")
    print(gm)

    # Problem 4.1(c): Measure the accuracy of the model by comparing it with the true distribution
    l1_dist = gm.compare("joint.dat")
    print("Problem 4.1(c): Measure the accuracy of the model by comparing it with the true distribution")
    print("The accuracy (L1 distance) of the graphical model is %f" % (l1_dist))

    # Problem 4.1(d): Query the model and compare with the true resutls
    print("Problem 4.1(d): Query the model and compare with the true resutls")
    cp1_bf = gm.query([1], {8 : 1, 11 : 1})
    cp1_td = gm.query([1], {8 : 1, 11 : 1}, "TrueDistribution", "joint.dat")
    print("L1 distance = %f" % (cp1_bf.diff(cp1_td)))
    
    cp2_bf = gm.query([7, 8, 9, 10, 11], {4 : 1})
    cp2_td = gm.query([7, 8, 9, 10, 11], {4 : 1}, "TrueDistribution", "joint.dat")
    print("L1 distance = %f" % (cp2_bf.diff(cp2_td)))

    cp3_bf = gm.query([10], {0 : 1})
    cp3_td = gm.query([10], {0 : 1}, "TrueDistribution", "joint.dat")
    print("L1 distance = %f" % (cp3_bf.diff(cp3_td)))

    # Problem 4.1(e): Forward sampling, reestimate and compare
    print("Problem 4.1(e): Forward sampling, reestimate and compare")
    sample_num = 1
    for add_sample_num in [9, 90, 900, 9000, 10000, 10000, 10000]: 
        sample_num += add_sample_num
        gm.sample("sample.dat", add_sample_num)
        gm.train("sample.dat")
        l1_dist = gm.compare("joint.dat")
        print("The accuracy (L1 distance) after %d sampling and reestimating is %f" % (sample_num, l1_dist))

