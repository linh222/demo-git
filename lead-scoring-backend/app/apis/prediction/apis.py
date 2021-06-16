from fastapi import APIRouter, Depends, status
from fastapi.openapi.models import APIKey

from app.api_key import get_api_key
from app.trainings.train import predict, validate_feature
from .schemas import FeatureModel, ResponseModel

router = APIRouter()


@router.post(
    "/v1/predict",
    status_code=status.HTTP_200_OK,
    response_model=ResponseModel,
)
async def predict_lead_score(feature: FeatureModel, api_key: APIKey = Depends(get_api_key)):
    model_version = '1.0'

    # transform input
    feature_input = [[
        feature.phone_type,
        feature.telco,
        feature.starting_09X,
        feature.starting_08X,
        feature.is_gmail,
        feature.is_yahoo_email,
        feature.is_educational_email,
        feature.package,
        feature.entity_lead_source,
        feature.os_version,
        feature.browser_type
    ]]

    feature_validated = validate_feature(feature_input, model_version=model_version)
    predicted_probability = predict(feature_validated, model_version=model_version)
    lead_score = float("{:.2f}".format(100 * predicted_probability[0]))

    return {
            "lead_id": feature.lead_id,
            "lead_score": lead_score,
            "model_version": model_version,
            "date": feature.date
        }


def include_router(app):
    app.include_router(router)
