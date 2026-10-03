def create_environmental_alert(
    risk: str,
    confidence: float,
    aqi: float,
) -> dict:
    if confidence < 0.60:
        return {
            "level": "REVIEW",
            "message": "Low-confidence prediction. Verify sensor data.",
        }
    if risk == "Dangerous" or aqi > 150:
        return {
            "level": "CRITICAL",
            "message": "Restrict exposure and escalate for engineering review.",
        }
    if risk == "Moderate" or aqi > 100:
        return {
            "level": "WARNING",
            "message": "Increase monitoring and investigate pollutant trend.",
        }
    return {
        "level": "NORMAL",
        "message": "Continue scheduled monitoring.",
    }