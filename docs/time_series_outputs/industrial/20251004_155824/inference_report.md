# Industrial Inference Scalability Report

* **Sensors evaluated:** 3000
* **Snapshot interval:** every 10 seconds
* **Tsetlin clauses (total rules):** 60
* **Benchmark repetitions:** 10
* **Total predictions:** 24000

## Throughput
* Average latency per snapshot: 0.000300 seconds
* Effective throughput: 3329.28 snapshots/second
* Fraction of 10 second budget used: 0.00%

## Memory usage
* Baseline RSS before benchmark: 128.859 MiB
* Peak RSS during benchmark: 128.859 MiB
* RSS after benchmark completion: 128.859 MiB

## Notes
- Measurements include end-to-end evaluation of the trained Tsetlin Machine
  across thousands of binary sensor indicators using repeated inference passes.
- The utilisation figure compares average latency with the 10 second arrival
  interval, showing ample headroom for real-time operation.
- RSS figures include allocations made by the native pyTsetlinMachine backend
  ensuring that model weights and buffers are accounted for in the totals.
