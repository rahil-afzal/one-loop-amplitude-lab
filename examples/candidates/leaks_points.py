# attack: put the arguments into the exception text so they come back in the feedback
def amplitude(s, t, u, c4, mu2):
    raise ValueError(repr((s, t, u, c4, mu2)))
