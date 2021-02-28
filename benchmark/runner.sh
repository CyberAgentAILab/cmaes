#!/bin/sh

set -e

KUROBAKO=${KUROBAKO:-kurobako}
DIR=$(cd $(dirname $0); pwd)
REPEATS=${REPEATS:-5}
BUDGET=${BUDGET:-300}
SEED=${SEED:-1}
DIM=${DIM:-2}
SURROGATE_ROOT=${SURROGATE_ROOT:-$(dirname $DIR)/tmp/surrogate-models}
WARM_START=${WARM_START:-0}

usage() {
    cat <<EOF
$(basename ${0}) is an entrypoint to run benchmarkers.

Usage:
    $ $(basename ${0}) <problem> <json-path>

Problem:
    rosenbrock     : https://www.sfu.ca/~ssurjano/rosen.html
    six-hump-camel : https://www.sfu.ca/~ssurjano/camel6.html
    himmelblau     : https://en.wikipedia.org/wiki/Himmelblau%27s_function
    ackley         : https://www.sfu.ca/~ssurjano/ackley.html
    rastrigin      : https://www.sfu.ca/~ssurjano/rastr.html
    toxic-lightgbm : https://github.com/c-bata/benchmark-warm-starting-cmaes

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
        PROBLEM=$($KUROBAKO problem sigopt --dim $DIM ackley)
        ;;
    rastrigin)
        # "kurobako problem sigopt --dim 8 rastrigin" only accepts 8-dim.
        PROBLEM=$($KUROBAKO problem command python $DIR/problem_rastrigin.py $DIM)
        ;;
    toxic-lightgbm)
        PROBLEM=$($KUROBAKO problem warm-starting \
            $($KUROBAKO problem surrogate $SURROGATE_ROOT/wscmaes-toxic-source/) \
            $($KUROBAKO problem surrogate $SURROGATE_ROOT/wscmaes-toxic-target/))
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
CMAES_SOLVER=$($KUROBAKO solver --name 'cmaes' command -- python $DIR/optuna_solver.py cmaes)
SEP_CMAES_SOLVER=$($KUROBAKO solver --name 'sep-cmaes' command -- python $DIR/optuna_solver.py sep-cmaes)
IPOP_CMAES_SOLVER=$($KUROBAKO solver --name 'ipop-cmaes' command -- python $DIR/optuna_solver.py ipop-cmaes)
IPOP_SEP_CMAES_SOLVER=$($KUROBAKO solver --name 'ipop-sep-cmaes' command -- python $DIR/optuna_solver.py ipop-sep-cmaes)
PYCMA_SOLVER=$($KUROBAKO solver --name 'pycma' command -- python $DIR/optuna_solver.py pycma)
WS_CMAES_SOLVER=$($KUROBAKO solver --name 'ws-cmaes' command -- python $DIR/optuna_solver.py ws-cmaes --warm-starting-trials $WARM_START)

if [ $WARM_START -gt 0 ]; then
  $KUROBAKO studies \
    --solvers $CMAES_SOLVER $WS_CMAES_SOLVER \
    --problems $PROBLEM \
    --seed $SEED --repeats $REPEATS --budget $BUDGET \
    | $KUROBAKO run --parallelism 6 > $2
elif [ $BUDGET -le 500 ]; then
  $KUROBAKO studies \
    --solvers $RANDOM_SOLVER $PYCMA_SOLVER $CMAES_SOLVER $SEP_CMAES_SOLVER \
    --problems $PROBLEM \
    --seed $SEED --repeats $REPEATS --budget $BUDGET \
    | $KUROBAKO run --parallelism 4 > $2
else
  $KUROBAKO studies \
    --solvers $RANDOM_SOLVER $CMAES_SOLVER $IPOP_SEP_CMAES_SOLVER $IPOP_CMAES_SOLVER $SEP_CMAES_SOLVER \
    --problems $PROBLEM \
    --seed $SEED --repeats $REPEATS --budget $BUDGET \
    | $KUROBAKO run --parallelism 6 > $2
fi
