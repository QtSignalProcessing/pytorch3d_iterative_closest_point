import warnings
import torch
from typing import List, Optional, Tuple, Union, NamedTuple, TYPE_CHECKING, Sequence
from collections import namedtuple
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import knn_cuda as _C
#from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union


def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded


class Pointclouds:
    _INTERNAL_TENSORS = [
        "_points_packed",
        "_points_padded",
        "_normals_packed",
        "_normals_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_cloud_idx",
        "_cloud_to_packed_first_idx",
        "_num_points_per_cloud",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(self, points, normals=None, features=None) -> None:
        self.device = torch.device("cpu")

        # Indicates whether the clouds in the list/batch have the same number
        # of points.
        self.equisized = False

        # Boolean indicator for each cloud in the batch.
        # True if cloud has non zero number of points, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of clouds)
        self._P = 0  # (max) number of points per cloud
        self._C = None  # number of channels in the features

        # List of Tensors of points and features.
        self._points_list = None
        self._normals_list = None
        self._features_list = None

        # Number of points per cloud.
        self._num_points_per_cloud = None  # N

        # Packed representation.
        self._points_packed = None  # (sum(P_n), 3)
        self._normals_packed = None  # (sum(P_n), 3)
        self._features_packed = None  # (sum(P_n), C)

        self._packed_to_cloud_idx = None  # sum(P_n)

        # Index of each cloud's first point in the packed points.
        # Assumes packing is sequential.
        self._cloud_to_packed_first_idx = None  # N

        # Padded representation.
        self._points_padded = None  # (N, max(P_n), 3)
        self._normals_padded = None  # (N, max(P_n), 3)
        self._features_padded = None  # (N, max(P_n), C)

        # Index to convert points from flattened padded to packed.
        self._padded_to_packed_idx = None  # N * max_P

        # Identify type of points.
        if isinstance(points, list):
            self._points_list = points
            self._N = len(self._points_list)
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)

            if self._N > 0:
                self.device = self._points_list[0].device
                for p in self._points_list:
                    if len(p) > 0 and (p.dim() != 2 or p.shape[1] != 3):
                        raise ValueError("Clouds in list must be of shape Px3 or empty")
                    if p.device != self.device:
                        raise ValueError("All points must be on the same device")

                num_points_per_cloud = torch.tensor(
                    [len(p) for p in self._points_list], device=self.device
                )
                self._P = int(num_points_per_cloud.max())
                self.valid = torch.tensor(
                    [len(p) > 0 for p in self._points_list],
                    dtype=torch.bool,
                    device=self.device,
                )

                if len(num_points_per_cloud.unique()) == 1:
                    self.equisized = True
                self._num_points_per_cloud = num_points_per_cloud
            else:
                self._num_points_per_cloud = torch.tensor([], dtype=torch.int64)

        elif torch.is_tensor(points):
            if points.dim() != 3 or points.shape[2] != 3:
                raise ValueError("Points tensor has incorrect dimensions.")
            self._points_padded = points
            self._N = self._points_padded.shape[0]
            self._P = self._points_padded.shape[1]
            self.device = self._points_padded.device
            self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)
            self._num_points_per_cloud = torch.tensor(
                [self._P] * self._N, device=self.device
            )
            self.equisized = True
        else:
            raise ValueError(
                "Points must be either a list or a tensor with \
                    shape (batch_size, P, 3) where P is the maximum number of \
                    points in a cloud."
            )

        # parse normals
        normals_parsed = self._parse_auxiliary_input(normals)
        self._normals_list, self._normals_padded, normals_C = normals_parsed
        if normals_C is not None and normals_C != 3:
            raise ValueError("Normals are expected to be 3-dimensional")

        # parse features
        features_parsed = self._parse_auxiliary_input(features)
        self._features_list, self._features_padded, features_C = features_parsed
        if features_C is not None:
            self._C = features_C

    def _parse_auxiliary_input(
        self, aux_input
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[int]]:
        """
        """
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):
            return self._parse_auxiliary_input_list(aux_input)
        if torch.is_tensor(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Points and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of points in each cloud."
                )
            if aux_input.device != self.device:
                raise ValueError(
                    "All auxiliary inputs must be on the same device as the points."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    points in a cloud."
            )

    def _parse_auxiliary_input_list(
        self, aux_input: list
    ) -> Tuple[Optional[List[torch.Tensor]], None, Optional[int]]:
        """
        """
        aux_input_C = None
        good_empty = None
        needs_fixing = False

        if len(aux_input) != self._N:
            raise ValueError("Points and auxiliary input must be the same length.")
        for p, d in zip(self._num_points_per_cloud, aux_input):
            valid_but_empty = p == 0 and d is not None and d.ndim == 2
            if p > 0 or valid_but_empty:
                if p != d.shape[0]:
                    raise ValueError(
                        "A cloud has mismatched numbers of points and inputs"
                    )
                if d.dim() != 2:
                    raise ValueError(
                        "A cloud auxiliary input must be of shape PxC or empty"
                    )
                if aux_input_C is None:
                    aux_input_C = d.shape[1]
                elif aux_input_C != d.shape[1]:
                    raise ValueError("The clouds must have the same number of channels")
                if d.device != self.device:
                    raise ValueError(
                        "All auxiliary inputs must be on the same device as the points."
                    )
            else:
                needs_fixing = True

        if aux_input_C is None:
            # We found nothing useful
            return None, None, None

        # If we have empty but "wrong" inputs we want to store "fixed" versions.
        if needs_fixing:
            if good_empty is None:
                good_empty = torch.zeros((0, aux_input_C), device=self.device)
            aux_input_out = []
            for p, d in zip(self._num_points_per_cloud, aux_input):
                valid_but_empty = p == 0 and d is not None and d.ndim == 2
                if p > 0 or valid_but_empty:
                    aux_input_out.append(d)
                else:
                    aux_input_out.append(good_empty)
        else:
            aux_input_out = aux_input

        return aux_input_out, None, aux_input_C

    def __len__(self) -> int:
        return self._N

    def __getitem__(
        self,
        index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor],
    ) -> "Pointclouds":
        """
        """
        normals, features = None, None
        normals_list = self.normals_list()
        features_list = self.features_list()
        if isinstance(index, int):
            points = [self.points_list()[index]]
            if normals_list is not None:
                normals = [normals_list[index]]
            if features_list is not None:
                features = [features_list[index]]
        elif isinstance(index, slice):
            points = self.points_list()[index]
            if normals_list is not None:
                normals = normals_list[index]


class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor

def is_pointclouds(pcl: Union[torch.Tensor, "Pointclouds"]) -> bool:
    return hasattr(pcl, "points_padded") and hasattr(pcl, "num_points_per_cloud")

def convert_pointclouds_to_tensor(pcl: Union[torch.Tensor, "Pointclouds"]):
    if is_pointclouds(pcl):
        X = pcl.points_padded()  # type: ignore
        num_points = pcl.num_points_per_cloud()  # type: ignore
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            X.shape[0],
            device=X.device,
            dtype=torch.int64,
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points


class ICPSolution(NamedTuple):
    converged: bool
    rmse: Union[torch.Tensor, None]
    Xt: torch.Tensor
    RTs: SimilarityTransform
    t_history: List[SimilarityTransform]


def iterative_closest_point(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    init_transform: Optional[SimilarityTransform] = None,
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    verbose: bool = False,
) -> ICPSolution:

    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = convert_pointclouds_to_tensor(Y)

    b, size_X, dim = Xt.shape

    if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
        raise ValueError(
            "Point sets X and Y have to have the same "
            + "number of batches and data dimensions."
        )

    if ((num_points_Y < Yt.shape[1]).any() or (num_points_X < Xt.shape[1]).any()) and (
        num_points_Y != num_points_X
    ).any():
        # we have a heterogeneous input (e.g. because X/Y is
        # an instance of Pointclouds)
        mask_X = (
            torch.arange(size_X, dtype=torch.int64, device=Xt.device)[None]
            < num_points_X[:, None]
        ).type_as(Xt)
    else:
        mask_X = Xt.new_ones(b, size_X)

    # clone the initial point cloud
    Xt_init = Xt.clone()

    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            assert (
                R.shape == torch.Size((b, dim, dim))
                and T.shape == torch.Size((b, dim))
                and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            ) from None
        # apply the init transform to the input point cloud
        Xt = _apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []

    # the main loop over ICP iterations
    for iteration in range(max_iterations):
        Xt_nn_points = knn_points(
            Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True
        ).knn[:, :, 0, :]

        # get the alignment of the nearest neighbors from Yt with Xt_init
        R, T, s = corresponding_points_alignment(
            Xt_init,
            Xt_nn_points,
            weights=mask_X,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection,
        )

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the root mean squared error
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)
        rmse = wmean(Xt_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                f"ICP iteration {iteration}: mean/max rmse = "
                + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            print(rmse_msg)

        # check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)

def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X

def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
        #  `Union[Tuple[int], int]`.
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
    #  `Union[Tuple[int], int]`.
    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )


def eyes(
    dim: int,
    N: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
   
    identities = torch.eye(dim, device=device, dtype=dtype)
    return identities[None].repeat(N, 1, 1)\
    

_KNN = namedtuple("KNN", "dists idx knn")


class _knn_points(Function):
    """
    Torch autograd Function wrapper for KNN C++/CUDA implementations.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        p1,
        p2,
        lengths1,
        lengths2,
        K,
        version,
        norm: int = 2,
        return_sorted: bool = True,
    ):
        """
        """
        if not ((norm == 1) or (norm == 2)):
            raise ValueError("Support for 1 or 2 norm.")

        idx, dists = _C.knn_points_idx(p1, p2, lengths1, lengths2, norm, K, version)

        # sort KNN in ascending order if K > 1
        if K > 1 and return_sorted:
            if lengths2.min() < K:
                P1 = p1.shape[1]
                mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
                # mask has shape [N, K], true where dists irrelevant
                mask = mask[:, None].expand(-1, P1, -1)
                # mask has shape [N, P1, K], true where dists irrelevant
                dists[mask] = float("inf")
                dists, sort_idx = dists.sort(dim=2)
                dists[mask] = 0
            else:
                dists, sort_idx = dists.sort(dim=2)
            idx = idx.gather(2, sort_idx)

        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        ctx.norm = norm
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        norm = ctx.norm
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()
        grad_p1, grad_p2 = _C.knn_points_backward(
            p1, p2, lengths1, lengths2, idx, norm, grad_dists
        )
        return grad_p1, grad_p2, None, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:
    """
    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, norm, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out

AMBIGUOUS_ROT_SINGULAR_THR = 1e-15
def corresponding_points_alignment(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = convert_pointclouds_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
            < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = wmean(Xt, weight=weights, eps=eps)
    Ymu = wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (dim + 1)
    ).any():
        warnings.warn(
            "Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return SimilarityTransform(R, T, s)
