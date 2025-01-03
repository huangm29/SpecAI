#!/bin/sh

cat <<EOF>tmp.py
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from STOUT import translate_reverse

print(translate_reverse(sys.argv[1]))
EOF

conda run -n STOUT python -W ignore tmp.py $1

rm tmp.py
