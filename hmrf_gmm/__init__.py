"""Spatial segmentation with multiple features using Hidden Markov Random Fields and Finite Mixture Models

Approach based on Wang et al. 2016 paper

"""


class Element(object):
    def __init__(self, pos, center):
        """Define Element structure"""
        self.pos = pos
        self.center = center
        # self.neighbors = neighbors
        # self.color = color
        # self.label = np.random.randint(1)
        # self.energy = energy
        # self.entropy = entropy
        # self.prob = prob
        self.neighbors = []
        # self.beta # correlation strength

    def __repr__(self):
        """Generate meaningful output of element"""
        s = "Element at pos %d " % self.pos
        if hasattr(self, "values"):
            s += "with values "
            for val in self.values:
                s += "%f, " % val
        if hasattr(self, "label"):
            s += "and current label %d" % self.label
        return s

    def update_label(self, clf):
        """Update label according to classifyer object"""
        self.label = clf.predict(np.array([[self.values[0]], [self.values[1]]]).T)


class Domain1D(object):

    def __init__(self):
        """Domain for 1-D HMRF-GMM clustering

        Idea: first start with this simple setup to test different implememntations, then
        extend to n-D (in space & feature space)!
        """
        pass




