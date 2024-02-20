import torch
# Trace of a tensor [1,k,k]
def _rank3_trace(x):
    return torch.einsum('ij->i', x)
def pump(s, adj): 
    k = s.size(-1)
    s = torch.tanh(s) # torch.Size([20, N, k]) One k for each N of each graph
    
    CT_num = _rank3_trace(torch.matmul(torch.matmul(s.t(),adj.to_dense()), s)) # Tr(S^T A S) 
    # Degree sparse
    degrees = adj.sum(dim=1).coalesce().values()
    deg = torch.diag(degrees).to_device(s.device)
    #deg = torch.sparse_coo_tensor(adj.coalesce().indices(), degrees, size=(adj.size(0), adj.size(0)), dtype=s.dtype, device=s.device)
    CT_den = _rank3_trace(torch.matmul(torch.matmul(s.t(), deg.to_dense()), s))  # Tr(S^T D S) 
    # Mask with adjacency if proceeds 
    CT_loss = -(CT_num / CT_den) # Tr(S^T A S) / Tr(S^T D S)
    CT_loss = torch.mean(CT_loss)
    # Orthogonality regularization.
    ss = torch.matmul(s.t(), s)  
    i_s = torch.eye(k).type_as(ss) 
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s )  
    ortho_loss = torch.mean(ortho_loss)
    return s, CT_loss, ortho_loss 