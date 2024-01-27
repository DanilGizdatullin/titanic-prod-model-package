from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    validated_data = pre_pipeline_preparation(data=input_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    Pclass: Optional[int]
    Name: Optional[str]
    Sex: Optional[str]
    Age: Optional[int]
    SibSp: Optional[int]
    Parch: Optional[int]
    Ticket: Optional[int]
    Fare: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
