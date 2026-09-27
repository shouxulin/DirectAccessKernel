# Compare reproduced results against the paper for fig10-13.
# Usage: bash compare.sh [run_suffix]
#   bash compare.sh        compares result/figN_paper.csv with result/figN.csv (reproduce.sh output)
#   bash compare.sh _ref   compares result/figN_paper.csv with result/figN_ref.csv
suffix=${1:-}

for fig in 10 11 12 13; do
    reported=./result/fig${fig}_paper.csv
    run=./result/fig${fig}${suffix}.csv
    echo "===== fig${fig}: ${reported} vs ${run}"
    if [ ! -f "${run}" ]; then
        echo "skip: ${run} not found"
    else
        python cal_diff.py "${reported}" "${run}"
    fi
    echo
done
