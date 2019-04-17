# marksim

Detects anomalies in unbalanced panel data using markov chain simulations. More about what is a panel data [here](https://en.wikipedia.org/wiki/Panel_data).


Let's say you have an unbalanced panel containing the growth rates of 1 mln. firms in a country in the last N years (could be any discrete periodic variable). Let's also suppose that you observe 10 firms that grow that fast that they are in the top 1% of growers each year. In other words, these firms grow very persistently. Think of google or amazon or any other successful company being on news.   

marksim helps to answer the following questions:

* Is this number 10 random? 
* Given the dynamics of firm's growth rates in the data, how many companies on average can we expect being so persistent?


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
python -m unittest discover -s src/tests
```

## Versioning

I use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/vvkorz/validpanda/tags).

## Authors

* **Vladimir Korzinov** - *Initial work* - [vvkorz](https://github.com/vvkorz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
