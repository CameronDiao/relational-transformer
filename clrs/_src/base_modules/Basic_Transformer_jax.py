from faulthandler import disable
import math

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Basic_RT(hk.Module):
    def __init__(
            self,
            nb_heads: int,
            graph_vec: str,
            disable_edge_updates: bool,
            node_vec_size: int,
            node_hid_size: int,
            edge_vec_size: int,
            edge_hid_size_1: int,
            edge_hid_size_2: int,
            dropout_rate: float,
            name: str = 'rt_aggr',
    ):
        super().__init__(name=name)

        self.graph_vec = graph_vec
        self.disable_edge_updates = disable_edge_updates

        self.H = nb_heads
        self.HS = node_vec_size // nb_heads
        self.NS = node_vec_size
        self.NHS = node_hid_size
        self.ES = edge_vec_size
        self.EHS1 = edge_hid_size_1
        self.EHS2 = edge_hid_size_2
        self.dropout_rate = dropout_rate

    def __call__(self, node_tensors, edge_tensors, graph_tensors, adj_mat, hidden, **unused_kwargs):
        block = RTTransformerLayer(self.graph_vec, self.disable_edge_updates, self.NS, self.H, self.HS,
                                   self.ES, self.NHS, self.EHS1, self.EHS2, self.dropout_rate,
                                   name=self.name)
        node_tensors, edge_tensors = block(node_tensors, edge_tensors, graph_tensors)

        return node_tensors, edge_tensors


class RTTransformerLayer(hk.Module):
    def __init__(
            self,
            graph_vec: str,
            disable_edge_updates: bool,
            NS: int,
            H: int,
            HS: int,
            ES: int,
            NHS: int,
            EHS1: int,
            EHS2: int,
            dropout_rate: float,
            name: str = 'rt_aggr'
    ):
        super().__init__(name=name)
        self.graph_vec = graph_vec
        self.disable_edge_updates = disable_edge_updates
        self.NS = NS
        self.H = H
        self.HS = HS
        self.ES = ES
        self.NHS = NHS
        self.EHS1 = EHS1
        self.EHS2 = EHS2
        self.dropout_rate = dropout_rate

    def __call__(self, node_tensors, edge_tensors, graph_tensors, **unused_kwargs):
        N = node_tensors.shape[1]
        NA = RTAttentionLayer(self.graph_vec, self.NS, self.H, self.HS, self.ES)
        NL1 = hk.Linear(self.NS)
        NLN1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        NL2 = hk.Linear(self.NHS)
        NL3 = hk.Linear(self.NS)
        NLN2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # residuals = NL1(NA(node_tensors, edge_tensors, graph_tensors))
        # residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
        # node_tensors = NLN1(node_tensors + residuals)

        attw_node_tensors = NA(node_tensors, edge_tensors, graph_tensors)
        residuals = NL1(attw_node_tensors)
        residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
        node_tensors = NLN1(node_tensors + residuals)

        residuals = NL3(jax.nn.relu(NL2(node_tensors)))
        residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
        node_tensors = NLN2(node_tensors + residuals)

        if not self.disable_edge_updates:
            source_nodes = jnp.expand_dims(node_tensors, 1)
            expanded_source_nodes = jnp.tile(source_nodes, (1, N, 1, 1))
            target_nodes = jnp.expand_dims(node_tensors, 2)
            expanded_target_nodes = jnp.tile(target_nodes, (1, 1, N, 1))
            reversed_edge_tensors = jnp.swapaxes(edge_tensors, -2, -3)
            input_tensors = (edge_tensors, reversed_edge_tensors, expanded_source_nodes, expanded_target_nodes)
            if self.graph_vec == 'att':
                global_tensors = jnp.expand_dims(graph_tensors, (1, 2))
                expanded_global_tensors = jnp.tile(global_tensors, (1, N, N, 1))
                input_tensors += (expanded_global_tensors,)

            concatenated_inputs = jnp.concatenate(input_tensors, axis=-1)

            EL1 = hk.Linear(self.EHS1)
            EL2 = hk.Linear(self.ES)
            ELN1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            EL3 = hk.Linear(self.EHS2)
            EL4 = hk.Linear(self.ES)
            ELN2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            residuals = EL2(jax.nn.relu(EL1(concatenated_inputs)))
            residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
            edge_tensors = ELN1(edge_tensors + residuals)

            residuals = EL4(jax.nn.relu(EL3(edge_tensors)))
            residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
            edge_tensors = ELN2(edge_tensors + residuals)

        return node_tensors, edge_tensors


class RTAttentionLayer(hk.Module):
    def __init__(self, graph_vec, NS, H, HS, ES):
        super().__init__()
        self.graph_vec = graph_vec
        self.NS = NS
        self.H = H
        self.HS = HS
        self.ES = ES

        self.scale = 1.0 / math.sqrt(HS)

    def separate_node_heads(self, x):
        new_shape = x.shape[:-1] + (self.H, self.HS)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def separate_edge_heads(self, x):
        new_shape = x.shape[:-1] + (self.H, self.HS)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 3, 1, 2, 4))

    def separate_graph_heads(self, x):
        x = jnp.expand_dims(x, -2)
        new_shape = x.shape[:-1] + (self.H, self.HS)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def concatenate_heads(self, x):
        x = jnp.transpose(x, (0, 2, 1, 3))
        new_shape = x.shape[:-2] + (self.NS,)
        return jnp.reshape(x, new_shape)

    def __call__(self, node_tensors, edge_tensors, graph_tensors):
        Wnq = hk.Linear(self.NS)
        Wnk = hk.Linear(self.NS)
        Wnv = hk.Linear(self.NS)

        Weq = hk.Linear(self.NS)
        Wek = hk.Linear(self.NS)
        Wev = hk.Linear(self.NS)

        if self.graph_vec == 'att':
            Wgq = hk.Linear(self.NS)
            Wgk = hk.Linear(self.NS)
            Wgv = hk.Linear(self.NS)

        B = node_tensors.shape[0]
        N = node_tensors.shape[1]
        H = self.H
        HS = self.HS

        eQ = Weq(edge_tensors)
        eK = Wek(edge_tensors)
        eV = Wev(edge_tensors)

        nQ = Wnq(node_tensors)
        nK = Wnk(node_tensors)
        nV = Wnv(node_tensors)

        if self.graph_vec == 'att':
            gQ = Wgq(graph_tensors)
            gK = Wgk(graph_tensors)
            gV = Wgv(graph_tensors)

        eQ = self.separate_edge_heads(eQ)
        eK = self.separate_edge_heads(eK)
        eV = self.separate_edge_heads(eV)

        nQ = self.separate_node_heads(nQ)
        nK = self.separate_node_heads(nK)
        nV = self.separate_node_heads(nV)

        if self.graph_vec == 'att':
            gQ = self.separate_graph_heads(gQ)
            gK = self.separate_graph_heads(gK)
            gV = self.separate_graph_heads(gV)

        if self.graph_vec == 'att':
            Q = eQ + jnp.reshape(nQ, (B, H, N, 1, HS)) + jnp.reshape(gQ, (B, H, 1, 1, HS))
            K = eK + jnp.reshape(nK, (B, H, 1, N, HS)) + jnp.reshape(gK, (B, H, 1, 1, HS))
        else:
            Q = eQ + jnp.reshape(nQ, (B, H, N, 1, HS))
            K = eK + jnp.reshape(nK, (B, H, 1, N, HS))
        Q = jnp.reshape(Q, (B, H, N, N, 1, HS))
        K = jnp.reshape(K, (B, H, N, N, HS, 1))
        QK = jnp.matmul(Q, K)
        QK = jnp.reshape(QK, (B, H, N, N))

        QK = QK * self.scale
        att_dist = jax.nn.softmax(QK, axis=-1)

        att_dist = jnp.reshape(att_dist, (B, H, N, 1, N))
        if self.graph_vec == 'att':
            v2 = eV + jnp.reshape(nV, (B, H, 1, N, HS)) + jnp.reshape(gV, (B, H, 1, 1, HS))
        else:
            v2 = eV + jnp.reshape(nV, (B, H, 1, N, HS))
        new_nodes = jnp.matmul(att_dist, v2)
        new_nodes = jnp.reshape(new_nodes, (B, H, N, HS))

        return self.concatenate_heads(new_nodes)
