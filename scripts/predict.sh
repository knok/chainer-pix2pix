#!/bin/sh

#!/bin/sh

python predict.py --out out --enc-npz models/enc_iter_354000.npz \
       --dec-npz models/dec_iter_354000.npz ./html/animal-861805__480.png
