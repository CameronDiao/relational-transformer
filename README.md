# Relational Attention: Generalizing Transformers for Graph-Structured Tasks

This is the official code for the [Relational Transformer (RT)](https://arxiv.org/abs/2210.05062) developed by
[Cameron Diao](https://github.com/CameronDiao) and [Ricky Loynd](https://github.com/rickyloynd-microsoft).
Our code is forked from the CLRS benchmark repository (version date: Sept. 6, 2022).

RT employs a mathematically elegant extension of transformer attention, which 
incorporates edge vectors as first-class model components. The RT class is defined in `_src/processors.py`, while relational attention is implemented in `_src/base_modules/Basic_Transformer_jax.py`.

## Installation

For proper package installation, please refer to the setup instructions in [the original CLRS README](https://github.com/deepmind/clrs/blob/b3bd8d1e912b5333964e0de49842a2019739fc53/README.md)

Alternatively, we provide an environment file (`environment.yml`) containing the packages used for our
experiments. Assuming you have Anaconda installed (including conda), you can create a virtual
environment using the environment file as follows:

```shell
cd relational-transformer
conda env create -f environment.yml
conda activate rt_clrs
```

## Running Experiments

The run file `run.py` is located in the `examples` directory.
`run.py` contains the code we used to obtain the main results of our paper, located in Table 19.

You can reproduce our results evaluating RT on the "Jarvis' March" task by running:

```shell
python3 -m clrs.examples.run
```

Or, for a short test run, you can specify:

```shell
python3 -m clrs.examples.run --train_items 320
```

Run hyperparameters are set in `run.py` and can be overridden on the command line. The hyperparameters
we provide can also be found in Table 2. Before changing any hyperparameters, you should note: 

* For any model using attention heads, the hidden size will be overwritten with the value
$\mathrm{nb}\textunderscore\mathrm{heads} \cdot \mathrm{head}\textunderscore\mathrm{size}$.

* In the original CLRS benchmark, certain algorithmic tasks have more validation and test
samples than others. We follow these task specifications, which are described in [the CLRS README](https://github.com/deepmind/clrs/blob/b3bd8d1e912b5333964e0de49842a2019739fc53/README.md)

* The current code does not support chunking during training.

## Citation

To cite the CLRS Algorithmic Reasoning Benchmark:

```latex
@misc{https://doi.org/10.48550/arxiv.2210.05062,
  doi = {10.48550/ARXIV.2210.05062},
  url = {https://arxiv.org/abs/2210.05062},
  author = {Diao, Cameron and Loynd, Ricky},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Relational Attention: Generalizing Transformers for Graph-Structured Tasks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Contributing

This project welcomes contributions and suggestions.
