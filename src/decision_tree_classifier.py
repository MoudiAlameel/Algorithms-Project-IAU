import numpy as np
from collections import Counter
from treenode import TreeNode
import math

class DecisionTree():
    """
    Hardcoded Decision Tree Classifier (using Information Gain/Entropy)
    [Greedy Algorithm Family]
    """

    def __init__(self, max_depth=5, min_samples_leaf=1,
                 min_information_gain=0.0, **kwargs) -> None:
        self.max_depth = max_depth 
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.labels_in_train = None
        self.tree = None
        self.feature_importances = None

    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))
    
    def _partition_entropy(self, subsets: list) -> float:
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2
        
    def _find_best_split(self, data: np.array) -> tuple:
        node_entropy_val = self._data_entropy(data[:, -1]) 
        min_part_entropy = 1e9
        
        # --- FIX 1: Initialize min split variables to None/empty arrays ---
        # This prevents the UnboundLocalError if the loop fails to run or fails to find a better split.
        g1_min, g2_min = np.array([]), np.array([])
        min_entropy_feature_idx = None
        min_entropy_feature_val = None
        
        feature_idx_to_use = list(range(data.shape[1]-1)) 

        for idx in feature_idx_to_use:
            # Check only percentile splits for efficiency
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            
            for feature_val in feature_vals:
                g1, g2, = self._split(data, idx, feature_val)
                
                if g1.shape[0] == 0 or g2.shape[0] == 0:
                    continue 

                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
                
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2
        
        # --- FIX 2: Handle the case where NO valid split was ever found (min_entropy_feature_idx is None) ---
        if min_entropy_feature_idx is None:
            # Return the original data as the "split" result, and set IG to 0.
            # This triggers the termination condition in _create_tree (via min_samples_leaf check).
            g1_min, g2_min = data, np.array([])
            min_part_entropy = node_entropy_val # Ensures IG is 0
        
        best_information_gain = node_entropy_val - min_part_entropy
        
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, best_information_gain


    def _find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:,-1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        for i, _ in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0] 
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        g1_data, g2_data, split_feature_idx, split_feature_val, information_gain = self._find_best_split(data)
        label_probabilities = self._find_label_probs(data)

        if current_depth > self.max_depth or \
           self.min_samples_leaf > g1_data.shape[0] or self.min_samples_leaf > g2_data.shape[0] or \
           information_gain < self.min_information_gain or \
           (information_gain == 0 and g1_data.shape[0] != 0 and g2_data.shape[0] != 0):
            return TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
            
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        current_depth += 1
        node.left = self._create_tree(g1_data, current_depth)
        node.right = self._create_tree(g2_data, current_depth)
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        node = self.tree
        while node.left or node.right:
            if node.feature_idx is None: break
                
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
                
            if node is None:
                return np.array([1/len(self.labels_in_train)] * len(self.labels_in_train)) 
        return node.prediction_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        Y_train_col = np.reshape(Y_train, (-1, 1)) 
        train_data = np.concatenate((X_train, Y_train_col), axis=1)

        self.tree = self._create_tree(data=train_data, current_depth=0)

        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            self.feature_importances = {k: v / total_importance for k, v in self.feature_importances.items()}

    def predict(self, X_set: np.array) -> np.array:
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        preds_int = np.argmax(pred_probs, axis=1)
        return preds_int
        
    def _calculate_feature_importance(self, node):
        if node is not None and node.feature_idx is not None:
            weight = node.data.shape[0] / self.tree.data.shape[0]
            self.feature_importances[node.feature_idx] += node.feature_importance * weight
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)