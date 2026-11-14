# SemRisk-SQL

Source snapshot for the 0.6B + 105-feature experiment. The frozen Qwen3-Embedding-0.6B scorer supplies candidate error probabilities. The risk layer discovers conjunctive rules, estimates conditional rule-group statistics, and ranks candidates by truncated-normal VaR.

The base scorer and risk layer each achieved 1,036 / 1,534 (67.54% EX) on the same official BIRD development candidate pool.

This snapshot contains the rule discovery, risk aggregation and semantic feature modules. Running the complete experiment requires the original scorer checkpoint, candidate and requirement caches, supporting FINER modules, and evaluation data.
