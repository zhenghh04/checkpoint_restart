import numpy as np
from scipy.optimize import root_scalar

def optimal_checkpoint_cadence(node_count, node_memory, MTBAI=0.67, chkpt_bandwidth="DAOS-128",
                               R_0=None):
    """
    Calculate optimal checkpointing cadence in hours for jobs of given size and memory-to-checkpoint/node
    
    This works by solving an updated version of Eq. (21) of Daly (2006) (see
    https://www.sciencedirect.com/science/article/pii/S0167739X04002213), where checkpointing
    time interval is replaced by checkpointing usage interval (node-hrs), required time
    for a checkpoint is replaced by required usage for a checkpoint (node-hrs), and
    mean time between failures (MTBF) is replaced by mean usage between application 
    interrupts (MUBAI, also in node-hrs). These replacements of times by usages are required
    so as to reflect the fact that failure probability is per node-hr, not just per hr.
    
    Parameters:
    
    node_count (int): Job size, i.e. number of nodes
    
    node_memory (float): Amount of memory per node to be checkpointed
    
    MTBAI (float, optional): Mean time between application interrupts, in hours, computed for 10624
      nodes (i.e. all of Aurora).
    
    chkpt_bandwidth (float or str, optional): Bandwidth to storage device. Either a float, in which
      case it is in units of GB/s, or "DAOS-128" (128-node DAOS, corresponding to 5000 GB/s), or
      "LUSTRE" (LUSTRE filesystem, corresponding to 650 GB/s)
      
    R_0 (float, optional): Application failure rate, in units of failures/node-hr. If this is specified
      it will override the value of MTBAI, even if MTBAI is also specified. The default MTBAI of 0.67
      hrs corresponds to R_0=1.4E-04 failures/node-hr
      
    Returns: checkpointing interval in hours.
    
    """
    
    if R_0 is None:
        R_0 = 1 / (MTBAI * 10624)
        
    if chkpt_bandwidth == "DAOS-128":
        chkpt_bandwidth = 5000.0
    elif chkpt_bandwidth == "LUSTRE":
        chkpt_bandwidth = 650.0
    elif not isinstance(chkpt_bandwidth, (float,int)):
        raise TypeError('chkpt_bandwidth must be float, "DAOS-128", or "LUSTRE"')
    
    # Checkpointing time tau_c, in hours
    tau_c = node_memory / (chkpt_bandwidth * 3600)
    
    # Usage for one checkpoint
    u_chk = tau_c * node_count**2
    
    # Expected failures while checkpointing
    z_chk = R_0 * u_chk
    
    # Function to be rooted
    def rootme(z_c, z_chk):
        res = np.exp(-z_c-z_chk) - (1-z_c)
        return res
    
    # Get the root
    res = root_scalar(rootme, args=(z_chk), bracket=(0,0.999))
    if not res.converged:
        raise RuntimeError("Root-find convergence failure")
    
    z_c = res.root
    
    # Computation usage between checkpoints
    u_c = z_c / R_0
    
    # Computation of time between checkpoints, in hrs
    t_c = u_c / node_count
    
    return t_c
    
    