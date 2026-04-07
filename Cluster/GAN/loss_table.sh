#!/bin/bash
# Parse Maia2 losses ? CSV (robust version)

LOG_FILE="${1:-st188776_maia2_train_19507.out}"
OUTPUT_FILE="${2:-losses_table.csv}"

echo "Parsing $LOG_FILE..."

# Write header
echo "chunk,total_loss,maia_loss,disc_loss" > "$OUTPUT_FILE"

awk -v out="$OUTPUT_FILE" '

# -------------------------------
# GAN-style lines
# Example:
# [250/5000] STGAN:... MAIA:... D:...
# -------------------------------
index($0,"STGAN:") && index($0,"MAIA:") {

    match($0, /\[[0-9]+/, a)
    chunk = substr(a[0],2)

    match($0, /STGAN:[0-9.\-]+/, b)
    stgan = substr(b[0],7)

    match($0, /MAIA:[0-9.\-]+/, c)
    maia = substr(c[0],6)

    match($0, /D:[0-9.\-]+/, d)
    disc = substr(d[0],3)

    printf "%s,%s,%s,%s\n", chunk, stgan, maia, disc >> out
}

# -------------------------------
# Simple MAIA-style lines
# Example:
# [10] Loss:6.839 MAIA:5.824
# -------------------------------
index($0,"Loss:") && index($0,"MAIA:") && !index($0,"STGAN:") {

    match($0, /\[[0-9]+/, a)
    chunk = substr(a[0],2)

    match($0, /Loss:[0-9.\-]+/, b)
    loss = substr(b[0],6)

    match($0, /MAIA:[0-9.\-]+/, c)
    maia = substr(c[0],6)

    printf "%s,%s,%s,\n", chunk, loss, maia >> out
}

' "$LOG_FILE"

COUNT=$(tail -n +2 "$OUTPUT_FILE" | wc -l)

echo "Found $COUNT training steps"
echo "Saved: $OUTPUT_FILE"

# -------------------------------
# Summary
# -------------------------------
if [[ $COUNT -gt 0 ]]; then
    echo ""
    echo "LAST 5:"
    tail -n 6 "$OUTPUT_FILE" | column -t -s,

    echo ""
    echo "AVERAGES:"
    awk -F, 'NR>1{
        sum_t+=$2;
        sum_m+=$3;
        if($4!="") sum_d+=$4;
        n++
    }
    END{
        print "Total:", sprintf("%.3f",sum_t/n);
        print "Maia: ", sprintf("%.3f",sum_m/n);
        if(sum_d>0) print "Disc: ", sprintf("%.3f",sum_d/n);
    }' "$OUTPUT_FILE"
fi
