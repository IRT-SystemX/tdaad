"""Window Functions."""

# Author: Martin Royer

from sklearn.pipeline import Pipeline


class LocalPipeline(Pipeline):
    """ Local pipeline modification for added functionality. """

    def __getitem__(self, ind):
        """ @hack to circumvent Pipeline's __getitem__ usage of self.__class__
        This allows LocalPipeline objects inheriting to still support slicing.
        It fails with their current implementation as self.__class__ constructor
        does not point to Pipeline constructor, but to the inherited class.

        Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return Pipeline(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            _, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est
