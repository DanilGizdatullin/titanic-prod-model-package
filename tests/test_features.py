from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

def test_extract_letter_transformer(sample_input_data):
    print(sample_input_data.head(20))
    transformer = ExtractLetterTransformer(variables=config.model_conf.cabin_vars)
    assert sample_input_data["Cabin"].iat[9] == "D47"

    subject = transformer.fit_transform(sample_input_data)

    assert subject["Cabin"].iat[9] == "D"
