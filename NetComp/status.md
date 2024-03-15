# NetComp Status and TODO

### October 13, 2017

## STATUS

Currently the major metrics are implemented. Resistance and lambda distances have been verified to be accurate; I have moderate confidence in the accuracy of DeltaCon and NetSimile metrics.

The next large and essential component of the package is the implementation of the fast (linear time) distances. I would also like to add more distances, as I learn of more useful ones.

## TODO

### Coding

- Code up fast resistance metric
- Code up fast DeltaCon metric


### Tests

- Test speed on all metrics. Use "sparse" graphs, i.e. graphs with order $n \log(n)$ edges. Probably just do ER graphs, for simplicity.
- Test accuracy on approximate metrics
    - fast vs full resistance
    - fast vs full deltacon
    - partial vs total eigenvalue comparison