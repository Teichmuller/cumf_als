
# GPU implementation of algothirms
## Smoothed Alternating Least Squares (SAS) for Large-Scale Smart Meter Data Imputation
Fork from [here](https://github.com/cuMF/cumf_als).

## Major Changes

* 7/21/2017
  * Change MaxIter from a constant to an argument
  * Removed unused code
  * Added comparison script to numpy implementation
* 7/9/2017
  * Added tested smoothing term
* 6/30/2017
  * Initiated repository
  * Added tested 64-bit floating point feature, which could be enabled by uncommenting USE_DOUBLE

### TODO
- [ ] Multi-GPU support
- [ ] Refactoring
