from typing import Tuple, Mapping

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .hvp_solver import HVPSolver
from .min_norm_solver import find_min_norm_element
from .linalg_solver import KrylovSolver, MINRESSolver, CGSolver


__all__ = ['KKTSolver', 'KrylovKKTSolver', 'CGKKTSolver', 'MINRESKKTSolver']


class KKTSolver(object):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            *,
            kkt_momentum: float = 0.0,
            create_graph: bool = False,
            grad_correction: bool = False,
        ) -> None:

        self.network = network
        self.hvp_solver = hvp_solver
        self.device = device
        self.kkt_momentum = kkt_momentum
        self.jacobians_momentum_buffer = None
        self.alphas_momentum_buffer = None
        self.create_graph = create_graph
        self.grad_correction = grad_correction


    def zero_grad(self) -> None:
        self.hvp_solver.zero_grad()


    def _jacobians_alphas_rhs(
            self,
            weights: Tensor,
            *,
            verbose: bool = True,
        ) -> Tuple[Tensor, Tensor, Tensor]:

        grad_correction = self.grad_correction
        kkt_momentum = self.kkt_momentum
        hvp_solver = self.hvp_solver

        jacobians = hvp_solver.grad(create_graph=self.create_graph)

        alphas, _ = find_min_norm_element(jacobians.detach())
        alphas = jacobians.new_tensor(alphas).detach()

        if verbose:
            print(jacobians.norm(dim=1).detach().cpu().numpy())
            if jacobians.size(0) == 2:
                cosine = jacobians[0].dot(jacobians[1]).div(jacobians[0].norm(2) * jacobians[1].norm(2)).item()
                angle = np.rad2deg(np.arccos(cosine))
                print(f'alphas={alphas},angle={angle}')
            else:
                print(f'alphas={alphas}')

        if grad_correction:
            alphas_jacobians = alphas.view(1, -1).matmul(jacobians).view(1, -1).detach()
            jacobians.sub_(alphas_jacobians)

        if kkt_momentum > 0.0:
            if self.alphas_momentum_buffer is None:
                self.alphas_momentum_buffer = torch.clone(alphas).detach()

            alphas_buf = self.alphas_momentum_buffer
            alphas_buf.mul_(kkt_momentum).add_(alphas, alpha=1 - kkt_momentum)
            alphas = alphas_buf

            jacobians_buf = self.jacobians_momentum_buffer
            jacobians_buf.mul_(kkt_momentum).add_(jacobians.detach(), alpha=1 - kkt_momentum)
            jacobians = jacobians_buf

        rhs = weights.view(1, -1).matmul(jacobians).view(-1)

        return jacobians, alphas, rhs.clone().detach()


    @torch.no_grad()
    def _print_alpha_beta_cosine(
            self,
            jacobians: Tensor,
            alphas: Tensor,
            direction: Tensor
        ) -> None:

        direction = self.hvp_solver.apply(direction, alphas)
        jacobians = jacobians.neg().detach()

        v1v1 = jacobians[0].dot(jacobians[0]).item()
        v1v2 = jacobians[0].dot(jacobians[1]).item()
        v2v2 = jacobians[1].dot(jacobians[1]).item()
        xv1 = direction.dot(jacobians[0]).item()
        xv2 = direction.dot(jacobians[1]).item()

        # (alpha * v1 + beta * v2 - x) * v1 = 0.
        # (alpha * v1 + beta * v2 - x) * v2 = 0.
        # alpha * v1v1 + beta * v1v2 = xv1
        # alpha * v1v2 + beta * v2v2 = xv2
        # J = v1v1 * v2v2 - 2 * v1v2
        # [v2v2, -v1v2] [xv1]
        # [-v1v2, v1v1] [xv2]
        # alpha = (v2v2 * xv1 - v1v2 * xv2) / J
        # beta = (xv2 * v1v1 - xv1 * v1v2) / J
        # J does not matter since we care about the cosine angle only, not the absolute difference.

        alpha = xv1 * v2v2 - xv2 * v1v2
        beta = xv2 * v1v1 - xv1 * v1v2
        total = abs(alpha) + abs(beta)
        alpha /= total
        beta /= total
        span = alpha * jacobians[0] + beta * jacobians[1]
        cosine = np.rad2deg(np.arccos(span.div(span.norm(2)).dot(direction.div(direction.norm(2))).item()))
        print(alpha, beta, cosine)


    def backward(
            self,
            weights: Tensor,
            *,
            verbose: bool = False,
        ) -> None:

        jacobians, alphas, rhs = self._jacobians_alphas_rhs(weights, verbose=verbose)
        direction = self._explore(jacobians, alphas, rhs, weights, verbose=verbose)
        self.apply_grad(direction, normalize=True)


    def _explore(
            self,
            jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            weights: Tensor,
            *,
            verbose: bool,
        ) -> Tensor:

        raise NotImplementedError


    @torch.no_grad()
    def cosine(self) -> float:
        jacobians, _ = self.hvp_solver.grad_batch(create_graph=False)
        cosine = jacobians[0].dot(jacobians[1]).div(jacobians[0].norm(2) * jacobians[1].norm(2)).item()
        return cosine


    @torch.no_grad()
    def apply_grad(
            self,
            direction: Tensor,
            *,
            normalize: bool = True,
        ) -> None:

        if normalize:
            direction.div_(direction.norm())
        offset = 0
        for p in self.hvp_solver.parameters:
            numel = p.numel()
            p.grad = direction[offset:offset + numel].view_as(p.data).clone()
            offset += numel
        assert offset == direction.size(0)


class KrylovKKTSolver(KKTSolver):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            krylov_solver: KrylovSolver,
            *,
            stochastic: bool = True,
            kkt_momentum: float = 0.0,
            create_graph: bool = False,
            grad_correction: bool = False,
        ) -> None:

        super(KrylovKKTSolver, self).__init__(
            network, hvp_solver, device,
            kkt_momentum=kkt_momentum,
            create_graph=create_graph,
            grad_correction=grad_correction,
        )

        self.stochastic = stochastic
        self.krylov_solver = krylov_solver

    def _explore(
            self,
            jacobians: Tensor,
            alphas: Tensor,
            rhs: Tensor,
            weights: Tensor,
            *,
            verbose: bool,
        ) -> Tensor:

        lazy_jacobians = None if self.stochastic else self.hvp_solver.grad_batch(create_graph=True)[0]
        with self.krylov_solver.solve(lazy_jacobians, jacobians, alphas, rhs, verbose=verbose) as results:
            direction, _ = results
        return direction


class CGKKTSolver(KrylovKKTSolver):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            *,
            stochastic: bool = True,
            kkt_momentum: float = 0.0,
            create_graph: bool = False,
            grad_correction: bool = False,
            tol: float = 1e-5,
            damping: float = 0.0,
            maxiter: int = 5,
            pd_strict: bool = True,
        ) -> None:

        krylov_solver = CGSolver(hvp_solver, device, tol, damping, maxiter, pd_strict)

        super(CGKKTSolver, self).__init__(
            network, hvp_solver, device, krylov_solver,
            stochastic=stochastic,
            kkt_momentum=kkt_momentum,
            create_graph=create_graph,
            grad_correction=grad_correction,
        )


class MINRESKKTSolver(KrylovKKTSolver):

    def __init__(
            self,
            network: nn.Module,
            hvp_solver: HVPSolver,
            device: torch.device,
            *,
            stochastic: bool = True,
            kkt_momentum: float = 0.0,
            create_graph: bool = False,
            grad_correction: bool = False,
            shift: float = 0.0,
            tol: float = 1e-5,
            damping: float = 0.0,
            maxiter: int = 50,
        ) -> None:

        krylov_solver = MINRESSolver(network, hvp_solver, device, shift, tol, damping, maxiter)

        super(MINRESKKTSolver, self).__init__(
            network, hvp_solver, device, krylov_solver,
            stochastic=stochastic,
            kkt_momentum=kkt_momentum,
            create_graph=create_graph,
            grad_correction=grad_correction,
        )
