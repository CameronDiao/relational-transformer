import abc
from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class Basic_GAT(hk.Module):
    def __init__(
            self,
            nb_layers: int,
            out_size: int,
            nb_heads: int,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gat_mp',
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        self.out_size = out_size
        self.nb_heads = nb_heads
        self.head_size = out_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
    ) -> _Array:

        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(bias_mat[..., None],
                            (1, 1, 1, self.nb_heads))  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)
        a_e = hk.Linear(self.nb_heads)
        a_g = hk.Linear(self.nb_heads)

        values = m(node_tensors)  # [B, N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        att_1 = jnp.expand_dims(a_1(node_tensors), axis=-1)
        att_2 = jnp.expand_dims(a_2(node_tensors), axis=-1)
        att_e = a_e(edge_tensors)
        att_g = jnp.expand_dims(a_g(graph_tensors), axis=-1)

        logits = (
                jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
                jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
                jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
                jnp.expand_dims(att_g, axis=-1)  # + [B, H, 1, 1]
        )  # = [B, H, N, N]
        coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
        ret = jnp.matmul(coefs, values)  # [B, H, N, F]
        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += skip(node_tensors)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ret = ln(ret)

        return ret


class Basic_GATv2(hk.Module):
    def __init__(
            self,
            nb_layers: int,
            out_size: int,
            nb_heads: int,
            mid_size: Optional[int] = None,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gatv2_aggr',
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        self.mid_size = mid_size
        self.out_size = out_size
        self.nb_heads = nb_heads
        self.head_size = out_size // nb_heads
        self.mid_head_size = self.mid_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
    ) -> _Array:

        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(bias_mat[..., None],
                            (1, 1, 1, self.nb_heads))  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        w_1 = hk.Linear(self.mid_size)
        w_2 = hk.Linear(self.mid_size)
        w_e = hk.Linear(self.mid_size)
        w_g = hk.Linear(self.mid_size)

        a_heads = []
        for _ in range(self.nb_heads):
            a_heads.append(hk.Linear(1))

        values = m(node_tensors)  # [B, N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        pre_att_1 = w_1(node_tensors)
        pre_att_2 = w_2(node_tensors)
        pre_att_e = w_e(edge_tensors)
        pre_att_g = w_g(graph_tensors)

        pre_att = (
                jnp.expand_dims(pre_att_1, axis=1) +  # + [B, 1, N, H*F]
                jnp.expand_dims(pre_att_2, axis=2) +  # + [B, N, 1, H*F]
                pre_att_e +  # + [B, N, N, H*F]
                jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
        )  # = [B, N, N, H*F]

        pre_att = jnp.reshape(
            pre_att,
            pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
        )  # [B, N, N, H, F]

        pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

        # This part is not very efficient, but we agree to keep it this way to
        # enhance readability, assuming `nb_heads` will not be large.
        logit_heads = []
        for head in range(self.nb_heads):
            logit_heads.append(
                jnp.squeeze(
                    a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
                    axis=-1)
            )  # [B, N, N]

        logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

        coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
        ret = jnp.matmul(coefs, values)  # [B, H, N, F]
        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += skip(node_tensors)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ret = ln(ret)

        return ret