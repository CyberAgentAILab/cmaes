#!/bin/sh

set -e

KUROBAKO=${KUROBAKO:-kurobako}
DIR=$(cd $(dirname $0); pwd)

usage() {
    cat <<EOF
$(basename ${0}) is an entrypoint to run benchmarkers.

Usage:
    $ $(basename ${0}) <problem> <json-path>

Problem:
    rosenbrock     : https://www.sfu.ca/~ssurjano/rosen.html
    six-hump-camel : https://www.sfu.ca/~ssurjano/camel6.html
    himmelblau     : https://en.wikipedia.org/wiki/Himmelblau%27s_function
    ackley         : Ackley function in https://github.com/sigopt/evalset

Options:
    --help, -h         print this

Example:
    $ $(basename ${0}) rosenbrock ./tmp/kurobako.json
    $ cat ./tmp/kurobako.json | kurobako plot curve --errorbar -o ./tmp
EOF
}

case "$1" in
    himmelblau)
        PROBLEM=$($KUROBAKO problem command python $DIR/problem_himmelblau.py)
        ;;
    rosenbrock)
        PROBLEM=$($KUROBAKO problem command python $DIR/problem_rosenbrock.py)
        ;;
    six-hump-camel)
        PROBLEM=$($KUROBAKO problem command python $DIR/problem_six_hump_camel.py)
        ;;
    ackley)
        PROBLEM=$($KUROBAKO problem sigopt --dim 10 ackley)
        ;;
    help|--help|-h)
        usage
        exit 0
        ;;
    *)
        echo "[Error] Invalid problem '${1}'"
        usage
        exit 1
        ;;
esac

RANDOM_SOLVER=$($KUROBAKO solver random)
CMAES_SOLVER=$($KUROBAKO solver --name 'cmaes' command python $DIR/optuna_solver.py cmaes)
IPOP_CMAES_SOLVER=$($KUROBAKO solver --name 'ipop' command python $DIR/optuna_solver.py cmaes)
PYCMA_SOLVER=$($KUROBAKO solver --name 'pycma' command python $DIR/optuna_solver.py pycma)

$KUROBAKO studies \
  --solvers $RANDOM_SOLVER $IPOP_CMAES_SOLVER $PYCMA_SOLVER $CMAES_SOLVER \
  --problems $PROBLEM \
  --seed 1 \
  --repeats 10 --budget 300 \
  | $KUROBAKO run --parallelism 5 > $2
