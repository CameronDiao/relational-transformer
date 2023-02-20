# Relational Attention: Generalizing Transformers for Graph-Structured Tasks

This is the official code for the [Relational Transformer (RT)](https://arxiv.org/abs/2210.05062) developed by
[Cameron Diao](https://github.com/CameronDiao) and [Ricky Loynd](https://github.com/rickyloynd-microsoft).

RT employs a mathematically elegant extension of transformer attention, which 
incorporates edge vectors as first-class model components.

## Installation

For proper package installation, please refer to the setup instructions in [the original CLRS README](https://github.com/deepmind/clrs/blob/b3bd8d1e912b5333964e0de49842a2019739fc53/README.md)

We also provide an environment file (`environment.yml`) containing the packages used for our
experiments. Assuming you have Anaconda installed (including conda), you can create a virtual
environment using the environment file as follows:

```shell
cd relational-transformer
conda env create -f environment.yml
conda activate rt_clrs
```

## Running Experiments

The run file `run.py` is located in the `src/examples` directory.
`run.py` contains the code we used to obtain the main results of our paper, located in Table 19.

You can reproduce our results evaluating RT on the "Jarvis' March" task by running:

```shell
python3 -m clrs.examples.run
```

The run will use hyperparameters set in `run.py`, which we report in Table 2. We further
note the following:

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

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.