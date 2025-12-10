signature_analysis_schema = {
    "name": "return_signature_analysis",
    "description": "Return structured comparison between reference and document signature images.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pixel_difference": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "region": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["region", "description"]
                }
            },
            "stroke_points": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reference_signature": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "stroke": {"type": ["integer", "null"]},
                                "start_point": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                        {"type": "string"}
                                    ]
                                },
                                "end_point": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                        {"type": "string"}
                                    ]
                                },
                                "curvature_points": {
                                    "type": "array",
                                    "items": {
                                        "oneOf": [
                                            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                            {"type": "string"}
                                        ]
                                    }
                                },
                                "description": {"type": "string"}
                            },
                            "required": ["start_point", "end_point", "curvature_points", "description"]
                        }
                    },
                    "document_signature": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "stroke": {"type": ["integer", "null"]},
                                "start_point": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                        {"type": "string"}
                                    ]
                                },
                                "end_point": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                        {"type": "string"}
                                    ]
                                },
                                "curvature_points": {
                                    "type": "array",
                                    "items": {
                                        "oneOf": [
                                            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                            {"type": "string"}
                                        ]
                                    }
                                },
                                "description": {"type": "string"}
                            },
                            "required": ["start_point", "end_point", "curvature_points", "description"]
                        }
                    },
                    "differences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["type", "description"]
                        }
                    }
                },
                "required": ["reference_signature", "document_signature", "differences"]
            }
        },
        "required": ["pixel_difference", "stroke_points"]
    }
}
