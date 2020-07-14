from contextlib import contextmanager
from functools import partial
from typing import Tuple

import numpy as np

from scipy.sparse.linalg import LinearOperator, minres

import torch
import torch.nn as nn
from torch import Tensor

from .hvp_solver import HVPSolver


__all__ = ['PDError', 'HVPLinearOperator', 'KrylovSolver', 'MINRESSolver', 'CGSolver']


class PDError(RuntimeError):
    pass


class HVPLinearOperator(LinearOperator):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            damping: float,
        ) -> None:

        shape = (hvp_solver.size, hvp_solver.size)
        dtype = list(network.parameters())[0].detach().cpu().numpy().dtype

        super(HVPLinearOperator, self).__init__(dtype, shape)

        self.network = network
        self.hvp_solver = hvp_solver
        self.device = device
        self.damping = damping

        self.jacobians = None
        self.alphas = None
        self.reset_parameters()

        self.hvp_counter = 0
        self.matvec_counter = 0
        self.reset_counters()


    def set_parameters(
            self,
            jacobians: Tensor,
            alphas: Tensor,
        ) -> None:

        self.jacobians = jacobians
        self.alphas = alphas


    def reset_parameters(self) -> None:
        self.jacobians = None
        self.alphas = None


    def reset_counters(self) -> None:
        self.hvp_counter = 0
        self.matvec_counter = 0


    def get_counters(self) -> Tuple[int, int]:
        return self.hvp_counter, self.matvec_counter


    def _matvec_tensor(
            self,
            tensor: Tensor,
        ) -> Tensor:

        alphas_hvps, _ = self.hvp_solver.apply(
            tensor, self.alphas, grads=self.jacobians, retain_graph=self.jacobians is not None) # (N,)
        if self.damping > 0.0:
            alphas_hvps.add_(tensor, alpha=self.damping)
        self.hvp_counter += 1
        self.matvec_counter += 1
        return alphas_hvps


    def _matvec(
            self,
            x: np.ndarray,
        ) -> np.ndarray:

        """HVP matrix-vector multiplication handler.

        If self is a linear operator of shape (N, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (N,) or (N, 1) ndarray.

        In our case, it computes alpha_hession @ x.
        """

        tensor = torch.as_tensor(x.astype(self.dtype), device=self.device)
        ret = self._matvec_tensor(tensor)
        return ret.detach().cpu().numpy()


class KrylovSolver(object):

    def solve(
            self,
            lazy_jacobians: Tensor,
            jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            *,
            verbose: bool = False,
        ) -> Tuple[Tensor, Tuple[int, int]]:

        raise NotImplementedError


class MINRESSolver(KrylovSolver):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            shift: float,
            tol: float,
            damping: float,
            maxiter: int,
        ) -> None:

        self.device = device
        self.linear_operator = HVPLinearOperator(network, hvp_solver, device, damping)
        self.minres = partial(minres, shift=shift, tol=tol, maxiter=maxiter)
        self.shape = self.linear_operator.shape
        self.dtype = self.linear_operator.dtype


    @contextmanager
    def solve(
            self,
            lazy_jacobians: Tensor,
            jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            *,
            verbose: bool = False,
        ) -> Tuple[Tensor, Tuple[int, int]]:

        """Control counters automatically.

        Parameters
        ----------
        lazy_jacobians : torch.Tensor or None
            If not None, it is for gradient reusing. A matrix with shape (M,N).
        jacobians : torch.Tensor
            A matrix with shape (M,N). It should be identical to `rhs` and
            `lazy_jacobians` in this case (if `lazy_jacobians` is not None).
        alphas: torch.Tensor
            An array with shape (M,).
        rhs: torch.Tensor
            A matrix with shape (N,).
        """

        try:
            self.linear_operator.set_parameters(lazy_jacobians, alphas)
            x0 = jacobians.mean(0).neg().clone().detach().cpu().numpy()
            rhs = rhs.cpu().numpy()
            results = self.minres(self.linear_operator, rhs, show=verbose, x0=x0)
            d = torch.as_tensor(results[0].astype(self.dtype), device=self.device)
            yield d, self.linear_operator.get_counters()
        finally:
            self.linear_operator.reset_parameters()
            self.linear_operator.reset_counters()


class CGSolver(KrylovSolver):

    def __init__(
            self,
            hvp_solver: HVPSolver,
            device: torch.device,
            tol: float,
            damping: float,
            maxiter: int,
            pd_strict: bool = False,
        ) -> None:

        self.hvp_solver = hvp_solver
        self.device = device
        self.tol = tol
        self.damping = damping
        self.maxiter = maxiter
        self.pd_strict = pd_strict

        self.hvp_counter = 0
        self.matvec_counter = 0
        self.reset_counters()


    def reset_counters(self) -> None:
        self.hvp_counter = 0
        self.matvec_counter = 0


    def cg(
            self,
            lazy_jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            x0: Tensor = None,
            *,
            verbose: bool = False,
        ) -> Tensor:

        hvp_solver_apply = self.hvp_solver.apply
        tol = self.tol
        damping = self.damping
        maxiter = self.maxiter
        pd_strict = self.pd_strict

        if x0 is None:
            x0 = torch.ones_like(rhs)
        x_next = x0.clone()

        r = hvp_solver_apply(x0, alphas, lazy_jacobians)
        r.add_(x0, alpha=damping).sub_(rhs)
        p = r.neg()
        r_k_norm = r.dot(r).item()

        if maxiter is None:
            n = len(rhs)
            maxiter = 2 * n
        for i in range(maxiter):
            Ap = hvp_solver_apply(p, alphas, lazy_jacobians).add(p, alpha=damping)
            pAp = p.dot(Ap).item()
            if pAp <= 0:
                if verbose:
                    print(i, round(pAp, 5), round(r_kplus1_norm, 5))
                if pd_strict:
                    if x0.dot(hvp_solver_apply(x0, alphas, lazy_jacobians).add(x0, alpha=damping)) <= 0:
                        raise PDError
                x_next.copy_(x0)
                break
            x0.copy_(x_next)
            alpha = r_k_norm / pAp
            x_next.add_(p, alpha=alpha)
            r.add_(Ap, alpha=alpha)
            r_kplus1_norm = r.dot(r).item()
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if verbose:
                print(i, round(pAp, 5), round(r_kplus1_norm, 5))
            if r_kplus1_norm < tol:
                break
            p = p.mul(beta).sub(r)
        return x_next


    def get_counters(self) -> Tuple[int, int]:
        return self.hvp_counter, self.matvec_counter


    @contextmanager
    def solve(
            self,
            lazy_jacobians: Tensor,
            jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            *,
            verbose: bool = False,
        ) -> Tuple[Tensor, Tuple[int, int]]:

        """Control counters automatically.

        Parameters
        ----------
        lazy_jacobians : torch.Tensor or None
            If not None, it is for gradient reusing. A matrix with shape (M,N).
        jacobians : torch.Tensor
            A matrix with shape (M,N). It should be identical to `rhs` and
            `lazy_jacobians` in this case (if `lazy_jacobians` is not None).
        alphas: torch.Tensor
            An array with shape (M,).
        rhs: torch.Tensor
            A matrix with shape (N,).
        """

        try:
            x0 = jacobians.mean(0).neg().clone().detach()
            d = self.cg(lazy_jacobians, alphas, rhs, x0, verbose=verbose)
            yield d, self.get_counters()
        finally:
            self.reset_counters()
