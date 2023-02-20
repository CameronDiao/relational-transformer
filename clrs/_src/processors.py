# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from clrs._src.base_modules.Basic_Transformer_jax import Basic_RT
from clrs._src.base_modules.Basic_MPNN_jax import Basic_MPNN
from clrs._src.base_modules.Basic_GAT_jax import Basic_GAT, Basic_GATv2

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class Processor(hk.Module):
    """Processor abstract base class."""

    @abc.abstractmethod
    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **kwargs,
    ) -> _Array:
        """Processor inference step.

        Args:
          node_fts: Node features.
          edge_fts: Edge features.
          graph_fts: Graph features.
          adj_mat: Graph adjacency matrix.
          hidden: Hidden features.
          **kwargs: Extra kwargs.

        Returns:
          Output of processor inference step.
        """
        pass

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False

class RT(Processor):
    def __init__(
            self,
            nb_layers: int,
            nb_heads: int,
            vec_size: int,
            node_hid_size: int,
            edge_hid_size_1: int,
            edge_hid_size_2: int,
            graph_vec: str,
            disable_edge_updates: bool,
            name: str = 'rt',
    ):
        super().__init__(name=name)
        assert graph_vec in ['att', 'core', 'cat']
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.graph_vec = graph_vec
        self.disable_edge_updates = disable_edge_updates

        self.node_vec_size = vec_size
        self.node_hid_size = node_hid_size
        self.edge_vec_size = vec_size
        self.edge_hid_size_1 = edge_hid_size_1
        self.edge_hid_size_2 = edge_hid_size_2
        self.global_vec_size = vec_size

        self.tfm_dropout_rate = 0.0

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        N = node_fts.shape[-2]
        node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, unused_kwargs.get('e_hidden')], axis=-1)
        if self.graph_vec == 'core':
            graph_tensors = jnp.concatenate([graph_fts, unused_kwargs.get('g_hidden')], axis=-1)
        else:
            graph_tensors = graph_fts

        if self.graph_vec == 'cat':
            expanded_graph_tensors = jnp.tile(jnp.expand_dims(graph_tensors, -2), (1, N, 1))
            node_tensors = jnp.concatenate([node_tensors, expanded_graph_tensors], axis=-1)
            expanded_graph_tensors = jnp.tile(jnp.expand_dims(graph_tensors, (-2, -3)), (1, N, N, 1))
            edge_tensors = jnp.concatenate([edge_tensors, expanded_graph_tensors], axis=-1)

        node_enc = hk.Linear(self.node_vec_size)
        edge_enc = hk.Linear(self.edge_vec_size)
        if self.graph_vec == 'core':
            global_enc = hk.Linear(self.global_vec_size)

        node_tensors = node_enc(node_tensors)
        edge_tensors = edge_enc(edge_tensors)
        if self.graph_vec == 'core':
            graph_tensors = global_enc(graph_tensors)
            expanded_graph_tensors = jnp.expand_dims(graph_tensors, 1)
            node_tensors = jnp.concatenate([expanded_graph_tensors, node_tensors], axis=-2)
            edge_tensors = jnp.pad(edge_tensors, [(0, 0), (1, 0), (1, 0), (0, 0)], mode='constant', constant_values=0.)

        layers = []
        for l in range(self.nb_layers):
            layers.append(Basic_RT(self.nb_heads,
                                   self.graph_vec,
                                   self.disable_edge_updates,
                                   self.node_vec_size,
                                   self.node_hid_size,
                                   self.edge_vec_size,
                                   self.edge_hid_size_1,
                                   self.edge_hid_size_2,
                                   self.tfm_dropout_rate,
                                   name='{}_layer{}'.format(self.name, l)))
        for layer in layers:
            node_tensors, edge_tensors = layer(node_tensors, edge_tensors, graph_tensors, adj_mat, hidden)

        if self.graph_vec == 'core':
            out_nodes = node_tensors[:, 1:, :]
            out_edges = edge_tensors[:, 1:, 1:, :]
            out_graph = node_tensors[:, 0, :]
        else:
            out_nodes = node_tensors
            out_edges = edge_tensors
            out_graph = graph_tensors

        return out_nodes, out_edges, out_graph if self.graph_vec == 'core' else None


class GAT(Processor):
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""

    def __init__(
            self,
            nb_layers: int,
            out_size: int,
            nb_heads: int,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gat_aggr',
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
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
            **unused_kwargs,
    ) -> _Array:
        """GAT inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        layers = []
        for l in range(self.nb_layers):
            layers.append(Basic_GAT(self.nb_layers,
                                    self.out_size,
                                    self.nb_heads,
                                    self.activation,
                                    self.residual,
                                    self.use_ln,
                                    name='{}_layer{}'.format(self.name, l)))
        for layer in layers:
            z = layer(z, edge_fts, graph_fts, adj_mat, hidden)

        return z, None, None


class GATFull(GAT):
    """Graph Attention Network with full adjacency matrix."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

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
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        self.head_size = out_size // nb_heads
        if self.mid_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the message!')
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
            **unused_kwargs,
    ) -> _Array:
        """GATv2 inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        layers = []
        for l in range(self.nb_layers):
            layers.append(Basic_GATv2(self.nb_layers,
                                      self.out_size,
                                      self.nb_heads,
                                      self.mid_size,
                                      self.activation,
                                      self.residual,
                                      self.use_ln,
                                      name='{}_layer{}'.format(self.name, l)))
        for layer in layers:
            z = layer(z, edge_fts, graph_fts, adj_mat, hidden)

        return z, None, None


class GATv2Full(GATv2):
    """Graph Attention Network v2 with full adjacency matrix."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGN(Processor):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    def __init__(
            self,
            nb_layers: int,
            out_size: int,
            mid_size: Optional[int] = None,
            mid_act: Optional[_Fn] = None,
            activation: Optional[_Fn] = jax.nn.relu,
            reduction: _Fn = jnp.max,
            msgs_mlp_sizes: Optional[List[int]] = None,
            use_ln: bool = False,
            name: str = 'mpnn_aggr',
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """MPNN inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        layers = []
        for l in range(self.nb_layers):
            layers.append(Basic_MPNN(self.nb_layers,
                                     self.out_size,
                                     self.mid_size,
                                     self.mid_act,
                                     self.activation,
                                     self.reduction,
                                     self._msgs_mlp_sizes,
                                     self.use_ln,
                                     name='{}_layer{}'.format(self.name, l)))
        for layer in layers:
            z = layer(z, edge_fts, graph_fts, adj_mat, hidden)

        return z, None, None


class DeepSets(PGN):
    """Deep Sets (Zaheer et al., NeurIPS 2017)."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, nb_nodes: int,
                 batch_size: int, **unused_kwargs) -> _Array:
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
    """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
    """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    @property
    def inf_bias(self):
        return True

    @property
    def inf_bias_edge(self):
        return True


class MemNetMasked(Processor):
    """Implementation of End-to-End Memory Networks.

    Inspired by the description in https://arxiv.org/abs/1503.08895.
    """

    def __init__(
            self,
            vocab_size: int,
            sentence_size: int,
            linear_output_size: int,
            embedding_size: int = 16,
            memory_size: Optional[int] = 128,
            num_hops: int = 1,
            nonlin: Callable[[Any], Any] = jax.nn.relu,
            apply_embeddings: bool = True,
            init_func: hk.initializers.Initializer = jnp.zeros,
            use_ln: bool = False,
            name: str = 'memnet') -> None:
        """Constructor.

        Args:
          vocab_size: the number of words in the dictionary (each story, query and
            answer come contain symbols coming from this dictionary).
          sentence_size: the dimensionality of each memory.
          linear_output_size: the dimensionality of the output of the last layer
            of the model.
          embedding_size: the dimensionality of the latent space to where all
            memories are projected.
          memory_size: the number of memories provided.
          num_hops: the number of layers in the model.
          nonlin: non-linear transformation applied at the end of each layer.
          apply_embeddings: flag whether to aply embeddings.
          init_func: initialization function for the biases.
          use_ln: whether to use layer normalisation in the model.
          name: the name of the model.
        """
        super().__init__(name=name)
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._linear_output_size = linear_output_size
        self._num_hops = num_hops
        self._nonlin = nonlin
        self._apply_embeddings = apply_embeddings
        self._init_func = init_func
        self._use_ln = use_ln
        # Encoding part: i.e. "I" of the paper.
        self._encodings = _position_encoding(sentence_size, embedding_size)

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """MemNet inference step."""

        del hidden
        node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                             axis=1)
        edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                                  ((0, 0), (0, 1), (0, 1), (0, 0)))
        nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                                   edge_fts_padded)

        # Broadcast hidden state corresponding to graph features across the nodes.
        nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
        return nxt_hidden

    def _apply(self, queries: _Array, stories: _Array) -> _Array:
        """Apply Memory Network to the queries and stories.

        Args:
          queries: Tensor of shape [batch_size, sentence_size].
          stories: Tensor of shape [batch_size, memory_size, sentence_size].

        Returns:
          Tensor of shape [batch_size, vocab_size].
        """
        if self._apply_embeddings:
            query_biases = hk.get_parameter(
                'query_biases',
                shape=[self._vocab_size - 1, self._embedding_size],
                init=self._init_func)
            stories_biases = hk.get_parameter(
                'stories_biases',
                shape=[self._vocab_size - 1, self._embedding_size],
                init=self._init_func)
            memory_biases = hk.get_parameter(
                'memory_contents',
                shape=[self._memory_size, self._embedding_size],
                init=self._init_func)
            output_biases = hk.get_parameter(
                'output_biases',
                shape=[self._vocab_size - 1, self._embedding_size],
                init=self._init_func)

            nil_word_slot = jnp.zeros([1, self._embedding_size])

        # This is "A" in the paper.
        if self._apply_embeddings:
            stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
            memory_embeddings = jnp.take(
                stories_biases, stories.reshape([-1]).astype(jnp.int32),
                axis=0).reshape(list(stories.shape) + [self._embedding_size])
            memory_embeddings = jnp.pad(
                memory_embeddings,
                ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
                 (0, 0), (0, 0)))
            memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
        else:
            memory = stories

        # This is "B" in the paper. Also, when there are no queries (only
        # sentences), then there these lines are substituted by
        # query_embeddings = 0.1.
        if self._apply_embeddings:
            query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
            query_embeddings = jnp.take(
                query_biases, queries.reshape([-1]).astype(jnp.int32),
                axis=0).reshape(list(queries.shape) + [self._embedding_size])
            # This is "u" in the paper.
            query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
        else:
            query_input_embedding = queries

        # This is "C" in the paper.
        if self._apply_embeddings:
            output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
            output_embeddings = jnp.take(
                output_biases, stories.reshape([-1]).astype(jnp.int32),
                axis=0).reshape(list(stories.shape) + [self._embedding_size])
            output_embeddings = jnp.pad(
                output_embeddings,
                ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
                 (0, 0), (0, 0)))
            output = jnp.sum(output_embeddings * self._encodings, 2)
        else:
            output = stories

        intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

        # Output_linear is "H".
        output_linear = hk.Linear(self._linear_output_size, with_bias=False)

        for hop_number in range(self._num_hops):
            query_input_embedding_transposed = jnp.transpose(
                jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

            # Calculate probabilities.
            probs = jax.nn.softmax(
                jnp.sum(memory * query_input_embedding_transposed, 2))

            # Calculate output of the layer by multiplying by C.
            transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
            transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

            # This is "o" in the paper.
            layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

            # Finally the answer
            if hop_number == self._num_hops - 1:
                # Please note that in the TF version we apply the final linear layer
                # in all hops and this results in shape mismatches.
                output_layer = output_linear(query_input_embedding + layer_output)
            else:
                output_layer = intermediate_linear(query_input_embedding + layer_output)

            query_input_embedding = output_layer
            if self._nonlin:
                output_layer = self._nonlin(output_layer)

        # This linear here is "W".
        ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

        if self._use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, None, None


class MemNetFull(MemNetMasked):
    """Memory Networks with full adjacency matrix."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_heads: Optional[int] = None,
                          **kwargs) -> ProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
      use_ln: Whether the processor passes the output through a layernorm layer.
      nb_heads: Number of attention heads for GAT processors.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """

    def _factory(out_size: int):
        if kind == 'deepsets':
            processor = DeepSets(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln
            )
        elif 'rt' in kind:
            processor = RT(
                nb_layers=kwargs['nb_layers'],
                nb_heads=nb_heads,
                vec_size=out_size,
                node_hid_size=kwargs['node_hid_size'],
                edge_hid_size_1=kwargs['edge_hid_size_1'],
                edge_hid_size_2=kwargs['edge_hid_size_2'],
                graph_vec=kwargs['graph_vec'],
                disable_edge_updates=kwargs['disable_edge_updates'],
                name=kind
            )
        elif kind == 'gat':
            processor = GAT(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln
            )
        elif kind == 'gat_full':
            processor = GATFull(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln
            )
        elif kind == 'gatv2':
            processor = GATv2(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln
            )
        elif kind == 'gatv2_full':
            processor = GATv2Full(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln
            )
        elif kind == 'memnet_full':
            processor = MemNetFull(
                vocab_size=out_size,
                sentence_size=out_size,
                linear_output_size=out_size,
            )
        elif kind == 'memnet_masked':
            processor = MemNetMasked(
                vocab_size=out_size,
                sentence_size=out_size,
                linear_output_size=out_size,
            )
        elif kind == 'mpnn':
            processor = MPNN(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln
            )
        elif kind == 'pgn':
            processor = PGN(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln
            )
        elif kind == 'pgn_mask':
            processor = PGNMask(
                nb_layers=kwargs['nb_layers'],
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln
            )
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _factory


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
    """Position Encoding described in section 4.1 [1]."""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)
