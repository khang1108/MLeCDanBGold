#!/usr/bin/env bash

set -euo pipefail

BUCKET="mlecdanbgold-hcmai-hk"
ROOT="data"

# Lấy toàn bộ folder cấp đầu tiên bên trong data/
aws s3api list-objects-v2 \
    --bucket "$BUCKET" \
    --prefix "$ROOT/" \
    --delimiter "/" \
    --query 'CommonPrefixes[].Prefix' \
    --output text |
tr '\t' '\n' |
while read -r prefix; do

    # data/L26_a/ -> L26_a
    folder="${prefix#$ROOT/}"
    folder="${folder%/}"

    # Screenshot của bạn có thể đang dùng Videos_L26_a.
    # Dòng này hỗ trợ cả:
    #   L26_a
    #   Videos_L26_a
    name="${folder#Videos_}"

    # Chỉ xử lý format L<number>_<suffix>
    if [[ "$name" =~ ^(L[0-9]+)_[a-zA-Z]+$ ]]; then

        group="${BASH_REMATCH[1]}"

        src="s3://$BUCKET/$ROOT/$folder/videos/"
        dst="s3://$BUCKET/$ROOT/$group/"

        echo
        echo "=============================================="
        echo "SOURCE      : $src"
        echo "DESTINATION : $dst"
        echo "=============================================="

        # PREVIEW ONLY
        aws s3 mv "$src" "$dst" \
            --recursive 
    fi
done