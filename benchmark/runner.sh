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
    six-hemp-camel : https://www.sfu.ca/~ssurjano/camel6.html
    himmelblau     : https://en.wikipedia.org/wiki/Himmelblau%27s_function

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
    six-hemp-camel)
        PROBLEM=$($KUROBAKO problem command python $DIR/problem_six_hemp_camel.py)
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
PYCMA_SOLVER=$($KUROBAKO solver --name 'pycma' command python $DIR/solver_pycma.py)
CMAES_SOLVER=$($KUROBAKO solver --name 'cmaes' command python $DIR/solver_cmaes.py)

$KUROBAKO studies \
  --solvers $RANDOM_SOLVER $PYCMA_SOLVER $CMAES_SOLVER \
  --concurrency 10 \
  --problems $PROBLEM \
  --seed 1 \
  --repeats 10 --budget 300 \
  | $KUROBAKO run --parallelism 5 > $2
