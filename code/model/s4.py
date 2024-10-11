# coding: utf-8

# NOTE: Borrowed and customized from https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
# import torch.nn.functional as F # <- Not used in the original .
# from einops import rearrange, repeat
# NOTE: To reduce dependent modules, einops are replaced with PyTorch's ops.

# from src.models.nn import DropoutNd
# NOTE: DropoutNd is borrowed from a different file and included at the bottom here.

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        # A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        A_imag = math.pi * torch.arange(N//2).unsqueeze(0).expand_as(log_A_real)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L, state=None):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        C = C.unsqueeze(0) # Dummy batch dim
        if not state is None: # NOTE: https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py#L1114C12-L1114C17
            s = state/dt.unsqueeze(-1)
            s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s,torch.ones_like(C)], dim=0)
            C = B * C
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L) NOTE: log of Vandermonde matrix
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('bhn, hnl -> bhl', C, torch.exp(K)).real

        if state is None:
            K_state = None
            K = K.squeeze(0)
        else:
            K_state = K[:-1] # State-dependent
            K = K[-1] # NOTE: B-dependent
        return K,K_state

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, state=None, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k,k_state = self.kernel(L=L, state=state) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        if not state is None:
            y = y + k_state

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
    
    def _setup_step(self):
        # NOTE: Borrowed from https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py#L1176.
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt.unsqueeze(-1) * A  # (H N)

        self.dA = torch.exp(dtA) # (H N)
        self.dB = B * (torch.exp(dtA)-1.) / A # (H N)
        self.dC = C

    def _get_params(self, rate=1.0):
        # NOTE: Borrowed and customized from https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py#L1074
        """Process the internal parameters."""

        # (S N) where S=n_ssm
        A = -torch.exp(self.kernel.log_A_real) + 1j * self.kernel.A_imag # Originally "- 1j" but must be 1j according to s4d.py.
        C = torch.view_as_complex(self.kernel.C) # (H N)
        B = torch.ones_like(C) # (H N)

        inv_dt = self.kernel.log_dt
        dt = torch.exp(inv_dt) * rate # (H N)

        return dt, A, B, C

# NOTE: DropoutNd below is borrowed and customized from https://github.com/state-spaces/s4/blob/main/src/models/nn/dropout.py

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            # if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            if not self.transposed: X = X.movedim(-1,1)
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            # if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            if not self.transposed: X = X.movedim(1,-1)
            return X
        return X