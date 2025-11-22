class TreeNode:
    """
    Represents a single node in the Decision Tree.
    Used by the DecisionTree class for tree construction and traversal.
    """
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
        # Data related to the split
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        
        # Metrics for feature importance calculation
        self.feature_importance = information_gain

        # Child nodes
        self.left = None
        self.right = None

    def node_def(self):
        """Returns a string definition of the node's split condition."""
        # Note: Feature names are not stored, only indices are used.
        return f"Feature[{self.feature_idx}] < {self.feature_val:.2f} (IG: {self.feature_importance:.4f})"