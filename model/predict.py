import pandas as pd
import shap
import uuid
from schemas.user_input import LoanApplication
from model.model_loder import ModelLoader
from utils.logger import get_logger

logger = get_logger()

pipeline = ModelLoader.load_model()

MODEL_VERSION = "1.0.0"

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

explainer = shap.Explainer(model)

def predict_output(data: LoanApplication):
   
    request_id = str(uuid.uuid4())
    try:

        logger.info(f"Request ID {request_id} - Prediction request received")
        input_data = data.model_dump()

        df = pd.DataFrame([input_data])

        df = df[pipeline.feature_names_in_]

        logger.info(f"Request ID {request_id} - Input dataframe prepared")
        prediction = int(pipeline.predict(df)[0])

        probabilities = pipeline.predict_proba(df)[0]

        confidence = float(max(probabilities))

        logger.info(f"Request ID {request_id} - Raw prediction: {prediction}")

   
        if prediction == 1:
            prediction_label = "Fully Paid"
        else:
            prediction_label = "Charged Off"

        logger.info(f"Request ID {request_id} - Prediction label: {prediction_label}")

    
        X_processed = preprocessor.transform(df)

        shap_values = explainer(X_processed)

        feature_names = preprocessor.get_feature_names_out()

        shap_list = shap_values.values[0]

        feature_importance = {
            feature: float(value)
            for feature, value in sorted(
                zip(feature_names, shap_list),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        }

        logger.info(f"Request ID {request_id} - SHAP explanation generated")

    
        return {

            "request_id": request_id,

            "model_version": MODEL_VERSION,

            "prediction": prediction_label,

            "default_probability": round(float(probabilities[0]), 4),

            "confidence": round(confidence, 4),

            "class_probabilities": {
                "Charged Off": round(float(probabilities[0]), 4),
                "Fully Paid": round(float(probabilities[1]), 4)
            },

            "top_features_influencing_prediction": feature_importance
        }

    except Exception as e:

        logger.error(f"Request ID {request_id} - Prediction failed: {str(e)}")

        return {
            "request_id": request_id,
            "error": str(e)
        }
