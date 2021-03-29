"""
Created on Sun Mar 21 16:53:38 2021

@author: haotian teng
"""
import networkx as nx
from networkx.algorithms.dag import topological_sort
from networkx.generators.random_graphs import random_regular_graph
import numpy as np
from math import gamma
from numpy.linalg import det
from numpy.random import choice
from typing import List,Union,Callable,Dict
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
from scipy.stats import multivariate_normal
from scipy.sparse import lil_matrix

import random

def graph_prior(min_D:int = 5,
                max_D:int = 20,
                alpha:float = 1.0)->List[float]:
    """
    Calculate the prior probabilities of graphs according to the maximum 
    in-dgree of the graph.
    D = max(indegree(G))
            1/S             if D<=min_D
    P(G) =  1/(alpha*D*S)   if min_D<=D<=max_D
            0               if D>max_D
    S is the normalization parameter.
    Parameters
    ----------
    G: nx.Graph
        A networkx Graph instance.    
    min_D : int, optional
        The low threshold of maximum indegree D. The default is 5.
    max_D : int, optional
        The up threshold of maximum indegree D. The default is 20.
    alpha : A shape parameter, higher the alpha, fast the decay of the
        prior probability.
    Returns
    -------
    List[float]
        The prior probabilities.
        
    """
    def _prior(G:nx.Graph):
        P_ = np.array([1 if x<=min_D else 1/alpha/x for x in np.arange(max_D)])
        P_ = P_/np.sum(P_)
        max_indegree = max(dict(G.in_degree).values())
        if max_indegree<max_D:
            return P_[max_indegree]
        else:
            return 0
    return _prior

class GaussianCausalNetwork(LinearGaussianBayesianNetwork):
    """
    A Gaussian Causal network is a bayesian network whose variables are
    all continuous and the conditional probability distribution (CPD) of the 
    variables are given by a Gaussian distribution on the linear combination
    of its parent nodes, the prior distribution of the parameters is chosen as
    Normal-inverse-gamma distribution by default, causal inference is achieved 
    by intervene, the class is inherited from the pgmpy.models.LinearGaussianBayesianNetwork.
    
    Reconstructing causal biological networks through active learning. Cho, H., et al.
    
        Parameters
    ----------
    G : nx.Graph
        The connection graph of the bayesian network, should be a DAG.
    weight_matrix : np.ndarray
        A N-by-(N+2) matrix represent the parameters, where N is the 
        number of the nodes in the graph, the first column is the mean and
        the last column is the standard deviation.
    node_list: List[str], optional.
        A list contains the name of the nodes. The default is None.
    structure_prior_function: Callable, optional.
        A function that returns the likelihood(Float) for a given graph.
    parameter_estimator:Func, optional.
        A function that 
    prior: Dict, optional.
        A dictionary contain the prior parameters:
            alpha: A length N vector. alpha in the inv-gamma distribution,
                  default is 1.0.
            beta: A length N vector. beta in the inv-gamma distribution,
                  default i 2.0.
            Lambda: A Nx(N+1)x(N+1) sparse matrix,covariance matrix in the 
                   normal-inverse-gamma distribution. Default is N identity
                   matrix.
            mu: A Nx(N+1) sparse matrix, the ith raw is the mean of prior
                normal-inverse-gamma distribution for node i. Default is zero
                matrix.
            
    """
    def __init__(self,
                 G:nx.Graph,
                 weight_matrix:np.ndarray,
                 node_list:List[str] = None,
                 structure_prior_function:Callable[[nx.Graph],float] = None,
                 # parameter_estimator:Estimator = None,
                 prior:Dict = None):

        super().__init__(G.edges)
        self.wm = weight_matrix
        self.node_n = len(G.nodes)
        self.node_list = node_list if node_list else sorted(G.nodes)
        self._graph = None
        self._adjacency = None
        if not structure_prior_function:
            self.structure_prior_f = lambda x: 1
        else:
            self.structure_prior_f = structure_prior_function
        self._op_map = {1:self._try_delete,
                        2:self._try_insert,
                        3:self._try_flip}
        self._revert_op_map = {1:self._revert_delete, 
                               2:self._revert_insert,
                               3:self._revert_flip}
        self.register_cpds()
        self.mu = lil_matrix((self.node_n,self.node_n+1))
        self.Lambda = [lil_matrix(np.eye(self.node_n+1))]*self.node_n
        self.alpha = np.ones(self.node_n)
        self.beta = np.ones(self.node_n)*2
    @property
    def weights(self):
        return self.wm
    
    @property
    def DAG(self):
        if self._graph is None:
            g = nx.DiGraph()
            g.add_nodes_from(sorted(self.nodes))
            g.add_edges_from(self.edges)
            self._graph = g
        return self._graph
    
    @DAG.setter
    def DAG(self,val):
        self._grpah = val

    @property
    def structure_prior(self):
        return self.structure_prior_f(self.DAG)
    
    @property
    def adjacency(self):
        if self._adjacency is None:
            self._adjacency = nx.adjacency_matrix(self.DAG)
        return self._adjacency
        
    @property
    def transpose_adj(self):
        return self.adjacency.transpose().tocsr()

    @property
    def non_edges(self):
        return list(nx.non_edges(self))
    
    def intervene_adj(self,
                      intervene_nodes:np.ndarray):
        adj = self.transpose_adj
        adj[intervene_nodes,:] = 0
        return adj

    
    
    def _clear_cache_properties(self):
        self._graph = None
        self._adjacency = None
    
    def register_cpds(self):
        cpds = []
        adjacency = self.transpose_adj
        for i in np.arange(self.node_n):
            idxs = adjacency[i].indices
            cpd = LinearGaussianCPD(i, 
                                    self.wm[i][np.concatenate(([0],idxs+1))], 
                                    self.wm[i][-1],
                                    list(idxs))
            cpd.variables = [*cpd.evidence, cpd.variable]
            cpds.append(cpd)
        self.add_cpds(*cpds)
        
    def get_efficent_weight(self):
        wm = self.wm.copy()
        wm[~self.adjacency] = 0
        return wm
    
    def update_cpd(self,
                   nodes:Union[List[int],int]):
        """
        Update the CPD of the given nodes.

        Parameters
        ----------
        nodes :Union[List[int],int]
            A list contains the index of the nodes to be updated.
        """
        if type(nodes) == int:
            nodes = [nodes]
        adjacency = self.transpose_adj
        for i in nodes:
            self.remove_cpds(self.get_cpds(i))
            idxs = adjacency[i].indices
            cpd = LinearGaussianCPD(i, 
                                    self.wm[i][np.concatenate(([0],idxs+1))], 
                                    self.wm[i][-1],
                                    list(idxs))
            cpd.variables = [*cpd.evidence, cpd.variable]
            self.add_cpds(cpd)
            
    def MHsampling_structure(self,
                             data:np.ndarray,
                             intervene_list:np.ndarray):
        """        
        Metropolis–Hastings sampling the graph structure based on given
        datasets and intervene records.

        Parameters
        ----------
        data : np.ndarray
            A T-by-N dataset, where T is the number of experiments, and N is
            the number of nodes.
        intervene_list : np.ndarray
            A T-by-N mask matrix, (i,j) indicates if the jth node has been 
            intervened in ith round.
        """
        old_logp = self.fit(data,intervene_list,update = False) + self.structure_prior
        op,edge = self._tweak()
        if edge is not None:
            e = edge
            if type(edge) == int:
                edge = list(self.edges)[edge]
            self.update_cpd(edge[1])
            if op == 3:
                self.update_cpd(edge[0])
            logp = self.fit(data,intervene_list,update = False) + self.structure_prior
            delta_logp = min(logp-old_logp,0)
            reject = random.random() > np.exp(delta_logp)
            if reject:
                self._revert_op_map[op](e)
                self._clear_cache_properties()
                self.update_cpd(edge[1])
                if op == 3:
                    self.update_cpd(edge[0])
            else:
                self.fit(data,intervene_list,update = True)
    
    def importance_sampling(self,
                            intervene_nodes:np.ndarray,
                            sample_n:int = 1):
        """
        A fast sampling on the G0 graph, i.e., the graph with no connecting 
        edge, and return the importance of this sampling (P(X|G)/P(X|G0))

        Parameters
        ----------
        intervene_nodes : np.ndarray
            A vector contains the indexs of the intervene nodes.
        sample_n: int
            The number of sampling times.
        Returns
        -------
        sample: A length N sample.
        importance: The importance score p/q.

        """
        intervene_n = len(intervene_nodes)
        mask = np.ones(self.node_n,dtype = bool)
        mask[intervene_nodes] = False
        mean = self.wm[mask,0]
        cov = np.diag(self.wm[mask,-1])
        samples = np.random.multivariate_normal(mean = mean,
                                               cov = cov,
                                               size = sample_n)
        residue = (samples-mean)
        q = -0.5*((residue@cov*residue).sum(axis = 1)+(self.node_n-intervene_n)*np.log(2*np.pi)+np.log(det(cov)))
        samples = np.insert(samples,intervene_nodes,0,axis = 1)
        p = [self.fit(x[None,:],
                     intervene_record = np.array([1-mask]),
                     update = False)
             for x in samples]
        return samples,np.exp(np.array(p)-q)
    
    def sampling(self,
                 intervene_nodes:np.ndarray,
                 intervene_values:np.ndarray,
                 n_sample:int = 1):
        """
        A direct sampling by the topological order of the network.

        Parameters
        ----------
        intervene_nodes : np.ndarray
            A vector contains the indexs of the intervene nodes.
        intervene_values : np.ndarray
            A vector contains the values of the intervene nodes.
        n_sample: int
            The number of sampling.
        Returns
        -------
        A length N sample vector.

        """
        assert len(intervene_nodes) == intervene_values.shape[0]
        X = np.zeros((n_sample,self.node_n))
        itv_mask = np.zeros(self.node_n,dtype=bool)
        itv_mask[intervene_nodes] = True
        X[:,itv_mask] = intervene_values
        for node in topological_sort(self.DAG):
            if node not in intervene_nodes:
                cpd = self.get_cpds(node)
                for X_i in X:
                    x = np.random.normal(loc = cpd.mean[0] + cpd.mean[1:]@X_i[cpd.evidence],
                                         scale = cpd.variance)
                    X_i[node] = x
        return X
    
    def fit(self,
            data:np.ndarray,
            intervene_record:np.ndarray,
            update:bool = True)->float:
        """
        Calculate the likelihood of the current graph structure.

        Parameters
        ----------
        data : np.ndarray
            A T-by-N matrix, where T is the total experiment number and N is 
            the number of nodes.
        intervene_record : np.ndarray
            A T-by-N mask matrix, (i,j) indicates if the jth node has been 
            intervened in ith round.
        update: bool
            If update the parameters.

        Returns
        -------
        logpdf : float
            The log likelihood of current graph.

        """
        ll = 0
        T = intervene_record.shape[0]
        intervene_record = intervene_record.astype(bool)
        no_int = np.invert(intervene_record)
        c = T*self.node_n - sum([sum(x) for x in intervene_record])
        ll+= -c/2*np.log(2*np.pi)
        a_ = self.alpha + self.node_n/2
        for i in np.arange(self.node_n):
            y = data[no_int[:,i],i] 
            if len(y) == 0:
                continue #skip if the node has no non-intervene data.
            pr_n = np.sum(self.transpose_adj[i,:])
            mask = np.repeat(self.transpose_adj[i,:].todense(),T,axis = 0).astype(bool) #T-by-N
            mask[intervene_record[:,i],:] = False
            p_mask = np.insert(self.transpose_adj[i,:].todense(),0,1,axis = 1)
            p_mask = np.array(p_mask)[0].astype(bool)
            pr_data = data[mask]
            if pr_data.shape[0] == 0:
                x = np.ones((no_int[:,i].sum(),1))
            else:
                x = np.insert(pr_data.reshape(-1,pr_n),0,1,axis = 1) #T-by-(N+1)
            Lambda = self.Lambda[i][p_mask,:][:,p_mask].todense() #Pr-by-Pr
            Lambda_ = Lambda + x.T@x
            mu = np.array(self.mu[i][:,p_mask].todense())[0] #Pr
            mu_ = np.array(np.linalg.inv(Lambda_)@(np.asarray(Lambda@mu)[0] + x.T@y))[0]
            alpha = self.alpha[i]
            alpha_ = a_[i]
            beta = self.beta[i]
            beta_ = beta + 0.5*(y.T@y+mu.T@Lambda@mu-mu_.T@Lambda_@mu_)
            ll += 0.5*(np.log(det(Lambda))-np.log(det(Lambda_)))+\
                alpha*np.log(beta) - alpha_*np.log(beta_)+\
                np.log(gamma(alpha_))-np.log(gamma(alpha))
            if update:
                self.wm[i,:-1][p_mask] = mu_
                self.wm[i,-1] = beta_/(alpha_+1)
                self.update_cpd([i])
        ll += np.log(self.structure_prior)
        return ll[0,0]
    def _tweak(self):
        operation = np.random.choice([1,2,3]) #Delete, insert and flip
        success = self._op_map[operation]()
        if success is not None:
            self._clear_cache_properties()
        return operation, success
        
    def _remove_edge(self,edge):
        self.remove_edge(*edge)
        return edge
        
    def _delete(self,edge_i = None):
        if edge_i is None:
            edge_n = len(self.edges)
            if edge_n == 0:
                raise ValueError("No edge in the graph to be deleted.")
            edge_i = np.random.randint(low = 0,high = edge_n)
        edge = list(self.edges)[edge_i]
        self.remove_edge(*edge)
        return edge
    
    def _insert(self,edge):
        self.add_edge(*edge)
        edge_i = list(self.edges).index(edge)
        return edge_i
        
    def _flip(self,edge_i):
        try:
            edge = list(self.edges)[edge_i]
            self._delete(edge_i)
            edge_i = self._insert(edge[::-1])
            return edge_i
        except ValueError:
            self.add_edge(*edge)
            return None
    
    def _try_delete(self,max_trail:int = 100)->Union[int,None]:
        """
        Try to delete an random edge into the graph.

        Parameters
        ----------
        max_trail : int, optional
            The maximum trail number. The default is 100.

        Returns
        -------
        edge_i : Union[int,None]
            If succeed, return the edge number in the new graph, otherwise
            return None.

        """
        edge_n = len(self.edges)
        if edge_n == 0:
            return None
        trail = 0
        while trail<max_trail:
            trail +=1
            try:
                edge_i = np.random.randint(low = 0,high = edge_n)
                edge = self._delete(edge_i)
                return edge
            except ValueError:
                pass
        return None
    
    def _try_insert(self,max_trail:int = 100)->Union[int,None]:
        """
        Try to insert an random edge into the graph.

        Parameters
        ----------
        max_trail : int, optional
            The maximum trail number. The default is 100.

        Returns
        -------
        edge_i : Union[int,None]
            If succeed, return the edge number in the new graph, otherwise
            return None.

        """
        trail = 0
        while trail<max_trail:
            edge = self.non_edges[choice(len(self.non_edges),1)[0]]
            trail +=1
            try:
                edge_i = self._insert(edge)
                return edge_i
            except ValueError:
                pass
        return None
                 
    def _try_flip(self,max_trail:int = 100)->Union[int,None]:
        """
        Try to flip an random edge in the graph.

        Parameters
        ----------
        max_trail : int, optional
            Maximum trail number. The default is 100.

        Returns
        -------
        Union[int,None]
            If succeed return the edge index, otherwise return None.

        """
        edge_n = len(self.edges)
        if edge_n == 0:
            return None
        trail = 0
        while trail<max_trail:
            trail +=1
            edge_i = np.random.randint(low = 0,high = edge_n)
            edge_i = self._flip(edge_i)
            if edge_i is not None:
                return edge_i
        return None
    
    def _revert_delete(self,edge):
        self._insert(edge)
    
    def _revert_insert(self,edge_i):
        self._delete(edge_i)
    
    def _revert_flip(self,edge_i):
        success = self._flip(edge_i)
        if success is None:
            raise ValueError("Failed to revert the flip operation.")
    
    def _efficent_weight_matrix(self):
        adjacency = nx.adjacency_matrix(self.G).todense()
        wm = self.wm[adjacency]
        return wm        
    
if __name__ == "__main__":
    ### Some basic operation of networkx
    MIN_DEGREE = 5
    N_GENE = 10
    SEED = 2021
    BURN_IN = 100
    GRAPH_N = 1
    EXP_N = 200
    np.random.seed(SEED)
    seeds = np.random.choice(np.arange(GRAPH_N),size = GRAPH_N,replace = False)
    print("Generate graphs.")
    Gs = [random_regular_graph(d = MIN_DEGREE, n = N_GENE, seed = np.random.RandomState(x)) for x in seeds]
    np.random.seed(SEED)
    print("Generate weights")
    weights = [np.random.rand(N_GENE,N_GENE+2) for _ in np.arange(GRAPH_N)]
    print("Build Bayesian networks.")
    GCNs = [GaussianCausalNetwork(G,weight_matrix = weight) for G,weight in zip(Gs,weights)]
    

        
