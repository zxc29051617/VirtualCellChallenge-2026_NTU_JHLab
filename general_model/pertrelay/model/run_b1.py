"""7-fold benchmark: consensus after the pipeline's shared removal vs AdaptNet with the full detail (beta 1), raw and
after the same removal."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B, adaptnet as AN
B.run("consensus_d200_shared10", lambda tr, h, t: B.consensus(tr, h, t, n0=200.0), post=B.post_shared)
fn = lambda tr, h, t: AN.method(tr, h, t, steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
B.run("adapt_b1_raw", fn)
B.run("adapt_b1_shared10", fn, post=B.post_shared)
