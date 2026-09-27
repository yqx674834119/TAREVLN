"""Ground VLM choices and independently verify claimed arrival."""
from .office_contract import validate_decision
from .office_tasks import TASKS
from .office_grounding import ground_visual_anchor
from .office_surface import verify_table_surface
import math


def visual_decision(api, prompt, images, context, views=()):
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
        verification_prompt = "Independently inspect only these current images. Do not infer hidden objects. Return JSON with boolean fields " + ", ".join(task["features"]) + ", inside_area, confidence (number 0 to 1), and short evidence. Also return anchor_view (front|left|rear|right) and anchor_pixel_xy [u,v]: the pixel at the centre of the VISIBLE TABLE TOP belonging to the requested furniture group, in the named 640 x 480 image. Coordinates are actual pixels (u: 0..639, v: 0..479), not normalized. Return null if that table is not visible. The requested area contains " + task["criteria"] + ". Ceiling lights, windows, and small PC monitors are not wall displays. " + task["arrival"] + " Judge CURRENT presence independently of any second-view requirement; a separate controller handles repeat observations. Do not propose movement."
        check = api.ask(verification_prompt, images, {"purpose": "arrival_verification", **context})
        grounding = ground_visual_anchor(check, views, context.get("position", []))
        if check.get("inside_area") is True and not grounding["valid"] and grounding["reason"] != "observed furniture too far":
            feedback = "\nThe RGB-D geometry rejected the proposed table pixel: " + str(grounding) + ". Recheck the current image and point to a visible TABLE TOP surface, not the floor, wall, or table legs. Use actual 640 x 480 pixel coordinates. If no tabletop is visible, return null. Return the complete JSON again."
            check = api.ask(verification_prompt + feedback, images, {"purpose": "depth_anchor_repair", **context})
            grounding = ground_visual_anchor(check, views, context.get("position", []))
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
        result["grounding"] = grounding
        surface = {'valid':False,'reason':'geometry or task features rejected'}
        if features and check['inside_area'] and grounding['valid']:
            surface = verify_table_surface(api,check,grounding,views,context)
        result['surface_identity'] = surface
        result["inside_target_area"] = features and check["inside_area"] and grounding["valid"] and surface['valid']
        if result["action"] == "confirm" and not result["inside_target_area"]:
            result["action"] = "explore"
            result["reason"] = "Fresh visual verification rejected arrival: " + str(check.get("evidence", ""))
    return result
