"""Ground VLM choices and independently verify claimed arrival."""
from .office_contract import validate_decision
from .office_tasks import TASKS
import math


def visual_decision(api, prompt, images, context):
    result = api.ask(prompt, images, {"purpose": "navigation_decision", **context})
    try:
        validate_decision(result, context["candidates"], context["candidate_view_ids"])
    except ValueError as error:
        result = api.ask(prompt + "\nYour previous reply failed a mechanical interface check: " + str(error) +
                         "\nReturn a corrected decision from the current images and the allowed ID-to-view mapping.",
                         images, {"purpose": "grounding_repair", **context})
        try:
            validate_decision(result, context["candidates"], context["candidate_view_ids"])
        except ValueError as second_error:
            # Invalid language output must not terminate the ROS stack or gain
            # motion authority. Resume native collision-aware exploration.
            return {"action": "explore", "candidate_id": None, "selected_view": None,
                    "target_visible": False, "inside_target_area": False, "confidence": 0.,
                    "evidence": "", "reason": "Rejected invalid VLM output after repair: " + str(second_error)}
    if result.get("target_visible") is True and float(result.get("confidence", 0)) >= .6:
        # A fresh visual check receives neither earlier claims nor the proposed
        # answer. This is still fallible VLM evidence, independently scored by
        # the private evaluator after the robot requests a stop.
        task = TASKS[context.get("task", "meeting")]
        check = api.ask("Independently inspect only these current images. Do not infer hidden objects. Return JSON with boolean fields " + ", ".join(task["features"]) + ", inside_area, confidence (number 0 to 1), and short evidence. The requested area contains " + task["criteria"] + ". Ceiling lights, windows, and small PC monitors are not wall displays. " + task["arrival"] + " Judge CURRENT presence independently of any second-view requirement; a separate controller handles repeat observations. Do not propose movement.", images,
                        {"purpose": "arrival_verification", **context})
        for field in [*task["features"], "inside_area"]:
            if type(check.get(field)) is not bool:
                raise ValueError("Invalid visual verification result")
        result["verification"] = check
        confidence = check.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid visual verification confidence")
        result["confidence"] = confidence
        features = all(check[k] for k in task["features"])
        result["target_visible"] = features
        result["inside_target_area"] = features and check["inside_area"]
        if result["action"] == "confirm" and not result["inside_target_area"]:
            result["action"] = "explore"
            result["reason"] = "Fresh visual verification rejected arrival: " + str(check.get("evidence", ""))
    return result
