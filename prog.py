import torch

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def remove_node(edges, node):
  edge1,edge2=[],[]
  for j in range(len(edges[0])):
    if edges[0][j]!=node and edges[1][j]!=node:
      edge1.append(edges[0][j])
      edge2.append(edges[1][j])
  return torch.tensor([edge1,edge2])

def deleted_neighborhood(data, element, radius):
  subset, edges, mapping, edge_mask= k_hop_subgraph(element,
                                                       radius,data.edge_index)
  return (subset[1:], remove_node(edges,element))

def aggregate(nodes, subset=None):
  if subset is None:
    if len(nodes)==0:
      return 0
    return sum(nodes)/len(nodes)
  if len(subset)==0:
    return 0
  return sum(nodes[subset])/len(subset)

import copy

import copy

def rnp(data,radii,subset=None,index=None):
  if len(radii)==0:
    return newData
  if subset is None:
    subset=data
    index=range(len(data))
  newData= copy.deepcopy(data)
  radius= radii[0]
  if len(radii)==1:
    for i in index:
      if len(subset.edge_index[0])!=0:
        nodes, edges = deleted_neighborhood(subset, i, radius)
      else:
        nodes= None
      newData.x[i]= aggregate(data.x,nodes)
    return newData
  for i in index:
    if len(subset.edge_index[0])!=0:
      nodes, edges= deleted_neighborhood(subset,i, radius)
    else:
       nodes= None
       edges= None
    tmp= rnp(data,radii[1:],Data(x=data.x[nodes],edge_index=edges),[int(i) for i in nodes])
    newData.x[i]= aggregate(tmp.x)
  return newData  