from torch.autograd import grad


def get_jjtvp(p, x):
    def jjtvp(v):
        v.requires_grad = True
        vj = grad(p, x, v, create_graph=True)[0]
        jjtv = grad(vj, v, vj.detach())[0]
        return jjtv
    return jjtvp


def get_jjtjjttvp(p, x):
    def jjtjjttvp(v):
        v.requires_grad = True
        vj = grad(p, x, v, create_graph=True)[0]
        jv = grad(vj, v, v, retain_graph=True)[0]
        vjj = grad(p, x, vj, retain_graph=True)[0]
        jjv = grad(vj, v, jv, retain_graph=True)[0]
        jtjv = grad(p, x, jv, retain_graph=True)[0]
        jjtv = grad(vj, v, vj, retain_graph=True)[0]
        jjtjjttv = jjtv - jjv - vjj + jtjv
        return jjtjjttv
    return jjtjjttvp


def get_hvp(g, x):
    def hvp(v):
        hv = grad(g, x, v, retain_graph=True)[0]
        return hv
    return hvp


def get_hdh1vp(g, x):
    def hdh1vp(v):
        hv = grad(g, x, v, retain_graph=True)[0]
        dh1v = v * grad(g.sum(), x, retain_graph=True)[0]
        hdh1v = hv - dh1v
        return hdh1v
    return hdh1vp
