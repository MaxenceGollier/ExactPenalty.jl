# Options Reference

Options that should be modified by expert users only are marked as *advanced*.

## Termination

* `atol::T = √eps(T)`: Desired convergence tolerance (absolute).

  >  

* `rtol::T = √eps(T)`: Desired convergence tolerance (relative).

  >

* `dual_inf_atol::T = √eps(T)`: TODO
* `dual_inf_rtol::T = √eps(T)`: TODO
* `primal_inf_atol::T = √eps(T)`: TODO
* `primal_inf_rtol::T = √eps(T)`: TODO


* `max_eval::Int = -1`: Maximum number of objective function evaluation.

  > 

* `max_time::Float64 = 30.0`: Maximum number of CPU seconds.

  > 

* `max_iter::Int = 10000`: Maximum number of *outer-loop* iterations.

  >

## Output

TODO

## NLP

TODO

## Initialization

TODO

## Penalty Parameter Update

TODO

## Quadratic Regularization

TODO

## Linear Solver

TODO

## Hessian Approximation