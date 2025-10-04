# Industrial Inference Scalability Report

* **Sensors evaluated:** 3000
* **Snapshot interval:** every 10 seconds
* **Tsetlin clauses (total rules):** 200
* **Benchmark repetitions:** 10
* **Total predictions:** 24000

## Throughput
* Average latency per snapshot: 0.001203 seconds
* Effective throughput: 831.34 snapshots/second
* Fraction of 10 second budget used: 0.01%

## Memory usage
* Peak RSS during inference loop (measured via tracemalloc): 0.020 MiB
* Current memory after benchmark: 0.000 MiB

## Notes
- Measurements include end-to-end evaluation of the trained Tsetlin Machine
  across thousands of binary sensor indicators using repeated inference passes.
- The utilisation figure compares average latency with the 10 second arrival
  interval, showing ample headroom for real-time operation.