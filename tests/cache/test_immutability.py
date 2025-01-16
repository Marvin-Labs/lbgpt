from lbgpt.cache import make_hash_chatgpt_request


def test_cache_immutability():
    messages = [
        {"role": "user", "content": "please respond with pong"},
    ]
    single_request_content = dict(
        messages=messages,
        model="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    res_base = "lbgpt_3d27b3f73fb10bf0b19074d827fe2630a89cf45760ea8472455797bbc000b512"

    for _ in range(20):
        res_next_pass = make_hash_chatgpt_request(single_request_content)
        assert res_base == res_next_pass


def test_cache_immutability_with_slack():
    single_request_content_1 = dict(
        messages=[{"role": "user", "content": "please respond with pong"}],
        model="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    single_request_content_2 = dict(
        messages=[
            {"role": "user", "content": "please respond with pong    "},
        ],
        model="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=5000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    assert make_hash_chatgpt_request(
        single_request_content_1
    ) == make_hash_chatgpt_request(single_request_content_2)
