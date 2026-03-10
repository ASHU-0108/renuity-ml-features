# Renuity ML Features

A pip-installable Python package for the Reunity project that acts as the single source of truth for ML feature engineering.

It takes a raw lead dictionary and enriches it in-place using pure Python operations, optimized for single-record millisecond-level API inference.

## Installation

Add the following line to your API repository's `requirements.txt` to install directly via Git using a Personal Access Token (PAT):

```text
# Azure DevOps / GitHub format Example
git+https://<YOUR_PAT_HERE>@github.com/YourOrganization/renuity-ml-features.git@main#egg=renuity-ml-features
```

*Note: Replace `<YOUR_PAT_HERE>` with your actual Personal Access Token and adjust the Git URL based on your provider.*

## Usage

```python
from renuity_ml_features import FeatureEngineer

# Example usage with your lookup dictionaries
enriched_lead_dict = FeatureEngineer(
    lead=my_raw_lead_dict,
    district_lookup=DISTRICT_LOOKUP_DICT,
    lp_source_lookup=LP_SOURCE_SUBSOURCE_LOOKUP_DICT,
    question_mapping=QUESTION_MAPPING_DICT,
).enrich()
```
