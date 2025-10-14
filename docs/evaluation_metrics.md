# MSD Benchmark: Evaluation Metrics

The **MSD (Multi-factor Sequential Disentanglement)** benchmark supports a flexible and extensible evaluation framework, centered around the `EvaluationManager` class. This module governs the evaluation process for trained models and coordinates key tools such as latent explorers, predictor-based evaluations, and judge models.

---

## EvaluationManager

The `EvaluationManager` is the core class that orchestrates model evaluation. It is responsible for:

- Loading models (from checkpoint or memory)
- Initializing the judge, predictor, and latent explorer
- Executing one or more evaluators across a dataset
- Repeating evaluations for robustness
- Logging results to the configured logger

### Key Features

- **Dataset agnostic**: Compatible with any data modality and format, across training, validation, and test sets.
- **Pluggable tools**: Judge, latent explorer, and predictors are configured at runtime.
- **Repeatable**: Supports multiple evaluation runs for computing mean and std scores.
- **Integration-ready**: Works seamlessly with training checkpoints and logger.

Each evaluation is performed by calling:

```python
evaluation_manager.evaluate(epoch=10)
```

For repeated testing (e.g., at test time):

```python
evaluation_manager.run_test()
```

This aggregates scores across `repeat` runs and logs average and standard deviation.

---

## AbstractEvaluator

Every evaluation metric in MSD is a subclass of `AbstractEvaluator`. Each evaluator is responsible for:

- Computing scores and summaries at a given epoch
- Logging results (summary `Series` and detailed `DataFrame`)
- Returning results to the evaluation manager

### Required Methods

```python
def eval(self, epoch: int) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Return a (summary, details) tuple.
    - summary: pd.Series with high-level scores (e.g., mean accuracy)
    - details: pd.DataFrame with per-class or per-dimension results
    """
```

```python
@property
def name(self) -> str:
    """
    Return the name of the evaluator.
    """
```

These allow the evaluator to report structured results.

---

## Built-in Evaluators

MSD includes several built-in evaluation tools, including:

- **Intervention-based (Two-/Multi-factor):** Measures how latent subsets influence specific ground-truth factors via controlled manipulations and classification by a judge model.
- **DCI Metrics:** Evaluates _Disentanglement_, _Completeness_, and _Informativeness_ of latent representations
- **Consistency:** Assesses the stability of static factor predictions under temporal or latent-space perturbations.

You can select which evaluators to use in the configuration under:

```yaml
evaluation:
  evaluators:
    - name: MultiFactorSample
      parameters: { }
    - name: LatentSwapVisualizer
      parameters: { }
```

### Judge Evaluators

Several evaluators in MSD rely on a **judge module** to assess correctness of factor predictions after latent manipulations. This includes:

- **Intervention-based evaluators** (e.g., MultiFactorSample, TwoFactorSample): use the judge to classify modified sequences and compare against expected factor changes.
- **Consistency evaluators**: rely on judge outputs to verify temporal and representational stability of static factor predictions.

Judges can be:
- A classifier trained specifically for the dataset (loaded via `classifier_loader_cfg`)
- A VLM-based judge using a pre-defined closed label set and (optionally) few-shot prompts

See [docs/configuration.md](configuration.md) for full usage examples.

---

## Custom Evaluators

To implement your own evaluation:

1. Inherit from `AbstractEvaluator`
2. Implement `eval(self, epoch)` to return a Series and DataFrame
3. Provide a `name` property

Example:

```python
class MyCustomEvaluator(AbstractEvaluator):
    def eval(self, epoch):
        """
        Return a (summary, details) tuple.
        - summary: pd.Series with high-level scores (e.g., mean accuracy)
        - details: pd.DataFrame with per-class or per-dimension results
        """
        summary = pd.Series({'score': 0.95})
        details = pd.DataFrame({'detail': [1, 2, 3]})
        return summary, details

    @property
    def name(self):
        """
        Return the name of the evaluator.
        """
        return "MyEvaluator"
```

---

## Summary

Evaluation in MSD is modular and designed for extensibility. The `EvaluationManager` coordinates the process, while individual metrics are handled by `AbstractEvaluator` subclasses. With built-in support for repeatable scoring and rich logging, the MSD benchmark ensures reliable and interpretable evaluations.