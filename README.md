cumule
======

Python implementation of Cumule algorithm

how to launch
-----
This command:
```
$ python silent_predictor.py --show_plots --evolution_period 3 --show_test_error --archive_threshold 0.001 100
```
Will run the algorithm for 100 timesteps, showing live plots, doing microbial selection every 3 timesteps, putting predictors in the archive if their train error is less than 0.001. It will also show a plot with test errors after desired number of steps. The rest of parameters(and there are many and even more to come) will be set to defaults(for example that means **no** replication. Plot in the end must look something like this:
![](http://imagizer.imageshack.us/v2/800x600q90/546/stmq.png)
It will also keep updating file *prediction.log* so I suggest executing `watch tail -n20 prediction.log` to get a better grasp of what's going on in there. Right now it's better than any graph, believe me.


Rest of the options(you can also run `python silent_predictor -h` to get this list):
```
usage: silent_predictor.py [-h] [-n NUM_PREDICTORS] [--runs RUNS]
                           [--epochs EPOCHS] [-ts TEST_SET_LENGTH]
                           [-e EVOLUTION_PERIOD] [-a ARCHIVE_THRESHOLD]
                           [-lr LEARNING_RATE] [-r] [-lg LOGFILE] [-i]
                           [--episode_length] [--show_test_error]
                           [--show_plots] [--sliding_training]
                           [--input_mutation_prob INPUT_MUTATION_PROB]
                           [--output_mutation_prob OUTPUT_MUTATION_PROB]
                           timelimit

positional arguments:
  timelimit

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_PREDICTORS, --num_predictors NUM_PREDICTORS
                        population size(default:50)
  --runs RUNS           number of runs(default:1)
  --epochs EPOCHS       number of epochs for each training(default:5)
  -ts TEST_SET_LENGTH, --test_set_length TEST_SET_LENGTH
                        test set length(default:50)
  -e EVOLUTION_PERIOD, --evolution_period EVOLUTION_PERIOD
                        evolution period(default:10)
  -a ARCHIVE_THRESHOLD, --archive_threshold ARCHIVE_THRESHOLD
                        threshold for getting into the archive(default: 0.02)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for predictors(default: 0.01)
  -r, --replication     enable weights replication(default: no)
  -lg LOGFILE, --logfile LOGFILE
                        log file name(default: prediction.log)
  -i, --mutate_input    enable input mask mutation(default: yes)
  --episode_length      number of samples per episode(default: 50)
  --show_test_error     test archive and show the plot
  --show_plots          show live plots
  --sliding_training    use sliding window of examples
  --input_mutation_prob INPUT_MUTATION_PROB
                        input mutation probability per bit(default: 0.05)
  --output_mutation_prob OUTPUT_MUTATION_PROB
                        output mutation probability per mask(default: 0.9)
```

known issues
-----
Anton(not an issue, just a name of the author of these words):
* There's a small chance you will see this error:
  ```
  ValueError: min() arg is an empty sequence
  ```
  This is a result of a lousy implementation of one of the less significant parts of algorithm. Will be fixed if proven important, until then just launch the code again - most likely it will go away.
* From my experiments, 50 steps is enough for convergence(filling entire archive) with archive threshold 0.01, and 100 for 0.001. Smaller number of steps are not thoroughly tested, so are not recommended for use. 
