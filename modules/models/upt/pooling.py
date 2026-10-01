import torch
import torch_scatter
from torch import nn
from modules.layers.embedding import ContinuousSincosEmbed
from modules.layers.cluster import get_ball_tree, flatten_ball_tree, get_center_of_balls
from einops import rearrange

class BallTreePartition:
    def __init__(self, num_balls=128):
        '''
        Partition the input geometry
        Args:
            num_balls: int, the number of balls to partition the input geometry into.
        '''
        self.num_balls = num_balls

    def partition_input(self, geometry):
        '''
        Partition the input geometry, in shape (1, num_points, 3), into a ball tree format, in shape (num_balls, num_points_per_ball, 3).
        '''
        geometry = geometry.squeeze()
        idx, mask = get_ball_tree(geometry[:, :3])

        len_tree = idx.shape[0]
        num_balls = self.num_balls
        assert len_tree % num_balls == 0, f"Tree size {len_tree} is not divisible by num_balls {num_balls}"
        num_points = len_tree // num_balls

        mask_at_level = rearrange(mask, '(n p) -> n p', n=num_balls, p=num_points)

        pc, ball_ids, offsets = flatten_ball_tree(geometry, idx, mask, mask_at_level)
        centers = get_center_of_balls(pc, ball_ids) # shape (num_balls, 3)

        return pc, centers, ball_ids, offsets

class BallTreePooling(nn.Module):
    """Ball-tree pooling: mean-aggregate per-point messages into `num_balls` ball tokens."""

    def __init__(
        self,
        hidden_dim: int,
        ndim: int,
        num_balls: int | None = None,
        mode: str = "relpos",
        coord_scale = 100,
        max_wavelength = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.mode = mode
        self.num_balls = num_balls
        self.coord_scale = coord_scale
        self.max_wavelength = max_wavelength

        self.partition = BallTreePartition(num_balls=num_balls)

        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim,
                                               ndim=ndim,
                                               max_wavelength=max_wavelength,)

        if mode == "abspos":
            message_input_dim = hidden_dim * 2
            self.rel_pos_embed = None
        elif mode == "relpos":
            message_input_dim = hidden_dim
            self.rel_pos_embed = ContinuousSincosEmbed(dim=hidden_dim,
                                                       ndim=ndim + 1,
                                                       max_wavelength=max_wavelength,)
        else:
            raise NotImplementedError

        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_dim = hidden_dim

    def create_messages(
        self,
        input_pos: torch.Tensor,
        dst_idx: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-point messages and the supernode positional embeddings.

        Args:
            input_pos: Tensor of shape (number_of_points, ndim), representing the coordinates of the input geometry.
            dst_idx: Ball index of each point in input_pos.
            centers: Tensor of shape (num_balls, ndim), representing the coordinate centers of the supernodes.

        Returns:
            (messages (number_of_points, hidden_dim), supernode embeddings (num_balls, hidden_dim)).
        """
        if self.mode == "abspos":
            x = self.pos_embed(self.coord_scale * input_pos)
            dst_embed = self.pos_embed(self.coord_scale * centers[dst_idx])
            x = torch.concat([x, dst_embed], dim=1)
        elif self.mode == "relpos":
            src_pos = input_pos # shape (number_of_points, ndim)
            dst_pos = centers[dst_idx] # shape (number_of_points, ndim)
            dist = dst_pos - src_pos
            mag = dist.norm(dim=1).unsqueeze(-1)
            x = self.rel_pos_embed(self.coord_scale * torch.concat([dist, mag], dim=1))
        else:
            raise NotImplementedError
        x = self.message(x)
        supernode_pos_embed = self.pos_embed(self.coord_scale * centers)

        return x, supernode_pos_embed

    @staticmethod
    def accumulate_messages(
        x: torch.Tensor,
        indptr: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of the messages between consecutive `indptr` offsets (one segment per ball)."""
        return torch_scatter.segment_csr(src=x, indptr=indptr, reduce="mean")

    def forward(
        self,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the ball-tree pooling layer.

        Args:
            input_pos: Tensor with shape (1, number_of_points_per_sample, ndim).

        Returns:
            Tuple of (x, centers), where x is the aggregated supernode features with shape
            (1, num_balls, hidden_dim) and centers are the ball centers with shape (num_balls, ndim).
        """
        pc, centers, ball_ids, offsets = self.partition.partition_input(input_pos)

        assert pc.shape[-1] == self.ndim, f"expected {self.ndim} channels, but got {pc.shape[-1]}"

        x, supernode_pos_embed = self.create_messages(
            input_pos=pc,
            dst_idx=ball_ids,
            centers=centers,
        )

        x = self.accumulate_messages(x, offsets)

        x = x.unsqueeze(0) # shape (1, num_balls, hidden_dim)

        # concatenate supernode pos embedding
        supernode_pos_embed = supernode_pos_embed.unsqueeze(0) # shape (1, num_balls, hidden_dim)

        x = torch.concat([x, supernode_pos_embed], dim=-1)
        x = self.proj(x)

        return x, centers
