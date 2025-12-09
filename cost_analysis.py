import tiktoken


def calculate_input_cost(token_count: int, cost_per_token: float = 2e-06) -> float:
    
    if token_count < 0:
        raise ValueError("Token count cannot be negative")

    cost = token_count * cost_per_token
    return round(cost, 3) 


def calculate_output_cost(token_count: int, cost_per_token: float = 8e-06) -> float:
    
    if token_count < 0:
        raise ValueError("Token count cannot be negative")

    cost = token_count * cost_per_token
    return round(cost, 3)


def calculate_faceapi_cost(num_calls: int):
    
    if num_calls < 0:
        raise ValueError("number of calls must be positive")

    total_cost = num_calls * 0.001

    return round(total_cost, 3)


def count_tokens(prompt: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(prompt))


def calculate_gemini_cost(num_calls: int):
    if num_calls < 0:
        raise ValueError("number of calls must be positive")
    
    total_cost = num_calls * 0.058

    return round(total_cost, 3)
