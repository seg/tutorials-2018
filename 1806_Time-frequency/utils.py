import os

class Manuscript:
    """Something to handle making figures with figure numbers.
    """
    def __init__(self, loc=None):
        self.loc = loc or ''
        self.fignum = 1
        self.figures = {}

    def savefig(self, figure, name):
        if not os.path.exists(self.loc):
            os.makedirs(self.loc)
        fname = self.figures.get(name)
        if fname is not None:
            figure.savefig(fname, dpi=300)
        else:
            f = f"figure_{self.fignum}.png"
            fname = self.figures.setdefault(name, os.path.join(self.loc, f))
            figure.savefig(fname, dpi=300)
            self.fignum += 1
        return
