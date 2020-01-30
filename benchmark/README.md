# Continuous benchmarking using kurobako and GitHub Actions

Benchmark scripts are built on [kurobako](https://github.com/sile/kurobako).
See [Introduction to Kurobako: A Benchmark Tool for Hyperparameter Optimization Algorithms](https://medium.com/optuna/kurobako-a2e3f7b760c7) for more details.

## How to run benchmark scripts

GitHub Actions continuously run the benchmark scripts and comment on your pull request.
If you want to run on your local machines, please execute following after installed kurobako.

```console
$ ./benchmark/runner.sh -h
runner.sh is an entrypoint to run benchmarkers.

Usage:
    $ runner.sh <problem> <json-path>

Problem:
    rosenbrock     : https://www.sfu.ca/~ssurjano/rosen.html
    six-hemp-camel : https://www.sfu.ca/~ssurjano/camel6.html
    himmelblau     : https://en.wikipedia.org/wiki/Himmelblau%27s_function

Options:
    --help, -h         print this

Example:
    $ runner.sh rosenbrock ./tmp/kurobako.json
    $ cat ./tmp/kurobako.json | kurobako plot curve --errorbar -o ./tmp

$ ./benchmark/runner.sh rosenbrock ./tmp/kurobako.json
$ cat ./tmp/kurobako.json | kurobako plot curve --errorbar -o ./tmp
```

`kurobako plot curve` requires gnuplot. If you want to run on Docker container, please execute following:

```
$ docker build -t cmaes ./benchmark
$ ./benchmark/himmelblau_runner.sh ./tmp/kurobako.json
$ docker run -it --rm -v $PWD/tmp:/volume cmaes
```

If you got something error, please investigate using:

```
$ docker run -it --rm -v $PWD/tmp:/volume --entrypoint sh cmaes
```

