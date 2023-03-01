# Some notes when going through
Basic steps:
1. Initialize the particles to original position `[0, 0, 0]`, and initialize the map using these particles.
2. Every formal step includes following three steps:
   1. Predict the poses of the particles of the next timestamp (here I adopted the encoder timeline as the basic timeline). In the prediction, we will add some noise to motion model in order to inject randomness to the process.
   2. Update the weights of the particles using the map correlation funciton. Then resample the particles according to the weights. The map correlation could be achieve by counting the number of matching obstacles. Check the effective number of the particles, and resample if it is lower than the threshold.
   3. Renew the occupancy log map using the most-likely particle. Then renew the obstacle map.