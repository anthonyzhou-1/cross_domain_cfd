import balltree
import torch

def length_to_offsets(lengths, device):
    """Converts a list of lengths to a list of offsets.
    lengths is a list of how long each segment is, for example: [3, 5, 2]
    offsets will be [0, 3, 8, 10] which means the first segment starts at 0 and ends at 3, and so on.

    Args:
        lengths: A list of lengths of shape (num_segments,).
        device: The device to create the tensor on (e.g., 'cpu' or 'cuda').
    Returns:
        A tensor of shape (num_segments + 1,) containing the offsets.
    """

    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def offsets_to_ids_tensor(offsets):
    '''Converts offsets to a tensor of segment ids.
    offsets is a tensor of shape (num_segments + 1), which contains the index of the start of each segment.
    For example, if offsets = [0, 3, 8, 10], then the segments are: [0:3], [3:8], [8:10].

    ids is a sorted tensor of shape (N, ) where N is the total number of elements across all segments, containing the segment id of each element. 
    For the example above, ids will be [0, 0, 0, 1, 1, 1, 1, 1, 2, 2].
    Args:
        offsets: A tensor of shape (num_segments + 1,) containing the offsets.
    Returns:
        ids: A tensor of shape (N,) where N is the total number of elements across all segments, containing the segment id of each element.
    '''

    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )

def get_ball_tree(pc):
    '''Gets the ball tree representation of the point ball.
    See: https://github.com/maxxxzdn/erwin/tree/main 
        - The ball tree is stored in memory contiguously - at each level of the tree, points in the same ball are stored next to each other
    The outputs are all in shape (2^max_level), where max_level is the maximum level of the ball tree.
    Since pc is not a power of 2, the storage format duplicates nodes (leaves) at the last level to make it a power of 2.

    Args:
        pc: A point ball tensor of shape (N, 3), where N is the number of points.
        
    Returns:
        tree_idx: A tensor of shape (2^max_level, ) containing the indices of the points in the ball tree.
        tree_mask: A tensor of shape (2^max_level, ) containing a mask indicating which points are valid in the ball tree.
    '''

    batch_idx = torch.zeros(len(pc), dtype=torch.int32, device=pc.device)
    tree_idx, tree_mask = balltree.build_balltree(pc, batch_idx)

    return tree_idx, tree_mask

def flatten_ball_tree(pc, tree_idx, tree_mask, mask_at_level):
    """Flatten the ball tree to a 1D representation
    Removes duplicates in the ball tree and returns a tensor of shape (N, 3) containing the points in the flattened ball tree.
    To index each ball, it also returns a tensor of shape (N, ) containing the ball ids for each point in the flattened ball tree.

    Args:
        pc: A point ball tensor of shape (N, 3), where N is the number of points.
        tree_idx: A tensor of shape (2^max_level, ) containing the indices of the points in the ball tree.
        tree_mask: A tensor of shape (2^max_level, ) containing a mask indicating which points are valid in the ball tree.
        mask_at_level: A tensor of shape (num_balls, padded_lengths) containing a mask indicating which points are valid in each ball.
    Returns:
        tree_pc: A tensor of shape (N, 3) containing the points in the flattened ball tree.
        ball_ids: A tensor of shape (N, ) containing the ball ids for each point in the flattened ball tree.
        offsets: A tensor of shape (num_balls + 1, ) containing the offsets for each ball.
    """
    tree_pc_idx = torch.masked_select(tree_idx, tree_mask) # shape (N, )
    tree_pc = pc[tree_pc_idx] # shape (N, 3)
    lengths = mask_at_level.sum(dim=-1) # different balls can have different # points
    offsets = length_to_offsets(lengths, device=tree_idx.device) # shape (num_balls + 1, )
    ball_ids = offsets_to_ids_tensor(offsets) # shape (N, )
    return tree_pc, ball_ids, offsets 

def get_center_of_balls(tree_pc, ball_ids):
    '''
    Computes the center of each ball in the ball tree.
    see https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html
    Args:
        tree_pc: A tensor of shape (N, 3) containing the points in the flattened ball tree.
        ball_ids: A tensor of shape (N, ) containing the ball ids for each point in the flattened ball tree.    
    Returns:
        A tensor of shape (num_balls, 3) containing the center of each ball.
    '''
    import torch_scatter 
    return torch_scatter.segment_coo(tree_pc, ball_ids.long(), reduce='mean')
