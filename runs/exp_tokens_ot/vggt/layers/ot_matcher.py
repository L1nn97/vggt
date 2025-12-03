import torch
from torch import nn
from torch.nn import functional as F
import ot

############################################
# The OT solvers
############################################

# 1
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int, lmba: float) -> torch.Tensor:
    """ 
    Perform Sinkhorn Normalization in Log-space for stability
    lambda: higher values result in lower entropy
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    Z = lmba*Z
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1)*lmba, dim=2)
        u = u/lmba
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2)*lmba, dim=1)
        v = v/lmba
    return Z + u.unsqueeze(2)*lmba + v.unsqueeze(1)*lmba

def log_optimal_transport(mu:torch.Tensor, nu:torch.Tensor, couplings: torch.Tensor, reg: float, sinkhorn_iterations: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    B, N_, M_ = couplings.shape
    log_mu, log_nu = mu.log(), nu.log()
    log_mu, log_nu = log_mu[None].expand(B, -1), log_nu[None].expand(B, -1)
    log_mu, log_nu = log_mu.to(couplings.device), log_nu.to(couplings.device)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, sinkhorn_iterations, 1/reg)
    # Z = Z - norm  # multiply probabilities by M+N
    return Z.exp()

# 2
def log_sinkhorn_iterations_kl(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int, lmba: float, reg_kl:float) -> torch.Tensor:
    """ 
    Perform Sinkhorn Normalization in Log-space for stability
    lambda: higher values result in lower entropy
    reg_kl: higher values result in stronger KL regularization
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    Z = lmba*Z
    phi= reg_kl/(reg_kl+1/lmba)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1)*lmba, dim=2)
        u = u/lmba * phi
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2)*lmba, dim=1)
        v = v/lmba * phi
    return Z + u.unsqueeze(2)*lmba + v.unsqueeze(1)*lmba

def log_optimal_transport_kl(mu:torch.Tensor, nu:torch.Tensor, couplings: torch.Tensor, reg: float, reg_kl: float, sinkhorn_iterations: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    B, N_, M_ = couplings.shape
    log_mu, log_nu = mu.log(), nu.log()
    log_mu, log_nu = log_mu[None].expand(B, -1), log_nu[None].expand(B, -1)
    log_mu, log_nu = log_mu.to(couplings.device), log_nu.to(couplings.device)
    Z = log_sinkhorn_iterations_kl(couplings, log_mu, log_nu, sinkhorn_iterations, 1/reg, reg_kl)
    # Z = Z - norm  # multiply probabilities by M+N
    return Z.exp()

# 3
def distributed_sinkhorn(couplings, reg, sinkhorn_iterations):

    Q = torch.exp(couplings / reg).transpose(-1,-2) # Q is N-by-M for consistency with notations from our paper
    N_1 = Q.shape[-2] # how many prototypes
    M_1 = Q.shape[-1] # number of samples to assign

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/N
        sum_of_rows = torch.sum(Q, dim=-1, keepdim=True)
        Q /= sum_of_rows
        Q /= N_1

        # normalize each column: total weight per sample must be 1/M
        Q /= torch.sum(Q, dim=-2, keepdim=True)
        Q /= M_1

    Q *= M_1 # the colomns must sum to 1 so that Q is an assignment
    return Q.transpose(-1,-2)


# 4
def ot_solver(a, b, couplings, type = "partial_wasserstein", reg = 0.005, reg_m_kl = 0.05, reg_m_l2 = 5):
    B, M, N = couplings.shape
    P_list = []
    for i in range(B):
        dist = 2-couplings[i]
        if type == "entropic":
            P = ot.sinkhorn(a, b, dist, reg)
        if type == "entropic_kl_uot":
            P = ot.unbalanced.sinkhorn_unbalanced(a, b, dist, reg, reg_m_kl)

        if type == "kl_uot":
            P = ot.unbalanced.mm_unbalanced(a, b, dist, reg_m_kl, div='kl')

        if type == "l2_uot":
            P = ot.unbalanced.mm_unbalanced(a, b, dist, reg_m_l2, div='l2')
        P_list.append(P)
    P = torch.stack(P_list)
    # if type == "partial_ot":
    #     P = ot.partial.partial_wasserstein(a, b, dist, m=alpha.item())
    #     P = 
    return P

############################################
# Problem definition, partial OT
############################################

def get_partial_ot_problem(scores, bin_score):
    if not isinstance(bin_score, torch.Tensor):
        bin_score = scores.new_tensor(bin_score)
    B, M, N = scores.shape
    dev = scores.device
    bins0 = bin_score.expand(B, M, 1).to(dev)
    bins1 = bin_score.expand(B, 1, N).to(dev)
    bin_score = bin_score.expand(B, 1, 1).to(dev)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, bin_score], -1)], 1)
    return couplings

############################################
# Problem definition, partial distributions that assign mass to all samples and (1-mass) to the bin
############################################

def get_gt_distributions(y_mat):
    # ground truth distribution
    mu = y_mat.sum(dim=1)/y_mat.sum() # distribution of prototypes
    nu = y_mat.sum(dim=0)/y_mat.sum() # distribution of features
    return mu, nu

def get_partial_distributions(N, M, mass):
    a, b = torch.ones((N,)) / N, torch.ones((M,)) / M  # uniform distribution on samples
    a = torch.cat([a*mass, a.new_tensor(1-mass)[None]])
    b = torch.cat([b*mass, b.new_tensor(1-mass)[None]])
    return a, b

def get_partial_distributions_input_marginals(prt, masses, mass_fg):
    mass_bkg = 1- mass_fg
    a = prt[0].flatten() * masses[0]/prt[0].sum()
    if a[a<1e-10].shape[0]>0:
        mass_bkg_a = torch.tensor(mass_bkg)
        a[a<1e-10] = mass_bkg_a/(a[a<1e-10].shape[0])
    else:
        mass_bkg_a = torch.tensor(0)
    # print(a[prt[0].flatten()>1e-10].sum())
    a_bin = (1 - (masses[0] + mass_bkg_a)).clone().detach()[None].to(a.device)
    a = torch.cat([a, a_bin])

    b = prt[1].flatten() * masses[1]/prt[1].sum()
    if b[b<1e-10].shape[0]>0:
        mass_bkg_b = torch.tensor(mass_bkg)
        b[b<1e-10] = mass_bkg_b/(b[b<1e-10].shape[0])
    else:
        mass_bkg_b = torch.tensor(0)
    # print(a[prt[0].flatten()<1e-10].sum())
    b_bin = (1 - (masses[1] + mass_bkg_b)).clone().detach()[None].to(b.device)
    b = torch.cat([b, b_bin])
    return a, b

def get_partial_log_distributions(N, M, mass):
    # specify the mass to be assigned
    one = torch.tensor(1.)
    ms, ns = (N*one), (M*one)
    # norm = - (ms + ns).log()
    # log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    # log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    mass = torch.tensor(mass)
    norm_m = - (ms).log() + mass.log()
    norm_n = - (ns).log() + mass.log()
    log_mu = torch.cat([norm_m.expand(N), (1-mass).log()[None]])
    log_nu = torch.cat([norm_n.expand(M), (1-mass).log()[None]])
    return log_mu, log_nu

############################################
# The Matcher Module with trainable bin score
############################################

def prob_add_bkg(prob_fg, mask0, mask1):
    # get the cosine similarity
    dev = prob_fg.device
    # n,m = prob_fg.shape
    n = mask0.shape[0]
    m = mask1.shape[0]
    # prepare the output, assign the probabilities to the foreground + bin
    mask_fg = torch.ones(n+1, m+1).to(dev)
    mask_fg[:-1,:-1][~mask0,:] = 0.0
    mask_fg[:-1,:-1][:,~mask1] = 0.0
    mask_fg[:-1,-1][~mask0] = 0.0
    mask_fg[-1,:-1][~mask1] = 0.0
    prob = torch.zeros(n+1, m+1).to(dev) #* -torch.inf
    prob[mask_fg.bool()] = prob_fg.flatten()
    prob[-1,-1] = 0 
    return prob

def ft_remove_bkg(ft0, ft1):
    th = 1e-8
    # get mask the background features
    mask0 = ft0.norm(dim=-1) > th
    mask1 = ft1.norm(dim=-1) > th
    
    ft0_fg = ft0[mask0]
    ft1_fg = ft1[mask1]
    return ft0_fg, ft1_fg, mask0, mask1

class SoftmaxMatcher(nn.Module):
    def __init__(self, sinkhorn_iterations=100, bin_score=0.4, mass=0.9, reg=0.1, reg_kl=0.01):
        super().__init__()
        # super(SoftmaxMatcher, self).__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        # bin_score = torch.nn.Parameter(torch.tensor(bin_score))# DINOv2 default value
        bin_score = torch.tensor(bin_score)
        # self.register_parameter('bin_score', bin_score)
        self.mass = mass
        self.reg = reg
        self.reg_kl = reg_kl

    def forward(self, ft0, ft1, y_mat=None, segmentations=None, masses=None):

        # remove bkg
        ft0_fg, ft1_fg, mask0, mask1 = ft_remove_bkg(ft0, ft1)

        # normalize the features (worse results!)
        ft0_fg = ft0_fg/ft0_fg.norm(dim=-1)[:,None]
        ft1_fg = ft1_fg/ft1_fg.norm(dim=-1)[:,None]

        cos_sim = torch.mm(ft0_fg, ft1_fg.t()) # N, M
        # Run the optimal transport on foreground features
        N, M = cos_sim.shape
        device = cos_sim.device

        # no gt y_mat available, we use the cosine similarity matrix
        if y_mat is not None:
            mu, nu = get_gt_distributions(y_mat)
        elif segmentations is not None and masses is not None:
            mu, nu = get_partial_distributions_input_marginals(segmentations, masses, self.mass)
        else:
            mu, nu = get_partial_distributions(N, M, mass=self.mass)
        couplings = get_partial_ot_problem(cos_sim[None], bin_score=self.bin_score)
        mu, nu = mu.to(device), nu.to(device)
        prob_out = log_optimal_transport_kl(mu, nu, couplings, reg=self.reg, reg_kl=self.reg_kl, sinkhorn_iterations=self.sinkhorn_iterations)[0] # diverging if reg_kl<1
        # prob_out = log_optimal_transport(mu, nu, couplings, reg=self.reg, sinkhorn_iterations=self.sinkhorn_iterations)[0]

        # add bkg
        prob = prob_add_bkg(prob_out, mask0, mask1)
        
        del prob_out, cos_sim
        return prob
    
    def get_cossim(self, ft0, ft1):
        ft0_fg, ft1_fg, mask0, mask1 = ft_remove_bkg(ft0, ft1)
        # normalize the features (worse results!)
        ft0_fg = ft0_fg/ft0_fg.norm(dim=-1)[:,None]
        ft1_fg = ft1_fg/ft1_fg.norm(dim=-1)[:,None]

        cos_sim = torch.mm(ft0_fg, ft1_fg.t())
        cos_sim = F.pad(cos_sim, (0,1,0,1), value=0)
        return cos_sim