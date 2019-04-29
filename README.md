# marksim

This project is work in progress. It's purpose for me is to learn some new things in python, c++ and devops.

marksim is a package that helps to detect anomalies in unbalanced panel data using markov chain simulations. More about what is a panel data [here](https://en.wikipedia.org/wiki/Panel_data).

The inspiration mainly comes from the following paper:

 - Henderson, A.D., Raynor, M.E. and Ahmed, M.: 2012, How long must a firm be great to rule out chance? benchmarking sustained superior performance without being fooled by randomness, *Strategic Management Journal* **33**, 387-406 [link](https://doi.org/10.1002/smj.1943)

The first version of this code was used in this [thesis](https://publikationen.bibliothek.kit.edu/1000084152). This repository contains a cleared and documented version of what I had initially.

## About
Let's say you have an unbalanced panel containing the growth rates of 1 mln. firms in a country in the last N years (could be any discrete periodic variable). Let's also suppose that you observe 10 firms that grow that fast that they are in the top 1% of growers each year. In other words, these firms grow very persistently. Think of google or amazon or any other successful company being on news. 

marksim helps to answer the following questions:

* Is this number 10 random? 
* Given the dynamics of firm's growth rates in the data, how many companies on average can we expect being so persistent?

## TODOs

here is my plan:

 - [x] python implementation of the calculation of transition probability matrix
 - [ ] python implementation of the simulation module
 - [ ] python implementation of the analytics module
 - [ ] release package on PyPI
 - [ ] docker image
 - [ ] django app with basic user interface
 - [ ] c++ implementation of simulations

## Usage

Package is a work in progress, so for now please install with

```
pip install .
```

For more detailed documentation, please see [here](https://marksim.readthedocs.io/en/latest/)

## Development

### Prerequisites

All dependencies are specified in requirements.txt file.

```
pip install -r requirements.txt
```

### Installing

```
pip install -e.
```

## Running tests

To run tests from the root folder do

```

```

## Versioning

I use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/vvkorz/validpanda/tags).

## Authors

* **Vladimir Korzinov** - *Initial work* - [vvkorz](https://github.com/vvkorz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
