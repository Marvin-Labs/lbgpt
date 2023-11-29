# Load Balancing ChatGPT (LBGPT)

Enhance your ChatGPT API experience with the LoadBalancing ChatGPT (LBGPT), a wrapper around OpenAI's API designed to boost performance, enable caching, and provide seamless integration with Azure's OpenAI API.

This tool significantly optimizes single request response times by asynchronously interacting with the OpenAI API and efficiently caching results. It also offers automatic retries in the event of API errors and the option to balance requests between OpenAI and Azure for an even more robust AI experience.

Proudly build by the team of [Marvin Labs](https://marvin-labs.com/) where we use AI to help financial analysts make better investment decisions.

## Installation
You can easily install LoadBalancing ChatGPT via pip:
```bash
pip install lbgpt
```

## Usage

### Basic
Initiate asynchronous calls to the ChatGPT API using the following basic example:

```python
import lbgpt
import asyncio

chatgpt = lbgpt.ChatGPT(api_key="YOUR_API_KEY")
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```

The `chat_completion_list` function expects a list of dictionaries with fully-formed OpenAI ChatCompletion API requests. Refer to the [OpenAI API definition](https://platform.openai.com/docs/api-reference/chat/create) for more details. You can also use the `chat_completion` function for single requests.

By default, LBGPT processes five requests in parallel, but you can adjust this by setting the `max_concurrent_requests` parameter in the constructor.



### Caching
Take advantage of request caching to avoid redundant calls:

```python
import lbgpt
import asyncio
import diskcache

cache = diskcache.Cache("cache_dir")
chatgpt = lbgpt.ChatGPT(api_key="YOUR_API_KEY", cache=cache)
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```

While LBGPT is tested with [diskcache](https://pypi.org/project/diskcache/), it should work seamlessly with any cache that implements the `__getitem__` and `__setitem__` methods.

### Azure
For users with an Azure account and proper OpenAI services setup, lbgpt offers an interface for Azure, similar to the OpenAI API. Here's how you can use it:

```python
import lbgpt
import asyncio

chatgpt = lbgpt.AzureGPT(api_key="YOUR_API_KEY", azure_api_base="YOUR AZURE API BASE", azure_model_map={"OPENAI_MODEL_NAME": "MODEL NAME IN AZURE"})
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```


You can use the same request definition for both OpenAI and Azure. To ensure interchangeability, map OpenAI model names to Azure model names using the `azure_model_map` parameter in the constructor (see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints for details).


### Load Balacing OpenAI and Azure
For optimal performance and reliability, it's recommended to set up the `LoadBalancedGPT` or `MultiLoadBalancedGPT`. These classes automatically balance requests between OpenAI and Azure, and they also offer caching and automatic retries.

`LoadBalancedGPT` offers load-balancing just between OpenAI and Azure models, but is slightly easier to set up. By default, 75% of requests are routed to the Azure API, while 25% go to the OpenAI API. You can customize this ratio by setting the `ratio_openai_to_azure` parameter in the constructor, taking into account that the Azure API is considerably faster.
  

```python
import lbgpt
import asyncio

chatgpt = lbgpt.LoadBalancedGPT(
    openai_api_key="YOUR_OPENAI_API_KEY",
    azure_api_key="YOUR_AZURE_API_KEY",
    azure_api_base="YOUR AZURE API BASE",
    azure_model_map={"OPENAI_MODEL_NAME": "MODEL NAME IN AZURE"})
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```

`MultiLoadBalancedGPT` offers load-balancing between multiple OpenAI and Azure models, and offers more flexibility in terms of the load balancing inputs. In order to achieve the same load balancing as the `LoadBalancedGPT`, you can use the following code:

```python
import lbgpt
import asyncio

openai_chatgpt = lbgpt.ChatGPT(api_key="YOUR_API_KEY")
azure_chatgpt = lbgpt.AzureGPT(api_key="YOUR_API_KEY", azure_api_base="YOUR AZURE API BASE", azure_model_map={"OPENAI_MODEL_NAME": "MODEL NAME IN AZURE"})


chatgpt = lbgpt.MultiLoadBalancedGPT(
    gpts=[openai_chatgpt, azure_chatgpt],
    allocation_function_weights=[0.25, 0.75],
    allocation_function='random',
)
    
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```

However, the MultiLoadBalancedGPT offers more flexibility in terms of the load balancing inputs, e.g. supporting multiple Azure instances or OpenAI keys. 

You can also select the allocation function `max_headroom` to automatically pick the API with the most available capacity. This requires you to tell the model constructors your RPM (requests per minute) and/or TPM (tokens per minute) limits. 

For example, if you have an OpenAI API key with a 5,000 TPM limit and an Azure API key with a 10,000 TPM limit, you can use the following code:

```python
import lbgpt
import asyncio

openai_chatgpt = lbgpt.ChatGPT(api_key="YOUR_API_KEY", limit_tpm=5_000)
azure_chatgpt = lbgpt.AzureGPT(api_key="YOUR_API_KEY", azure_api_base="YOUR AZURE API BASE", azure_model_map={"OPENAI_MODEL_NAME": "MODEL NAME IN AZURE"}, limit_tpm=10_000)


chatgpt = lbgpt.MultiLoadBalancedGPT(
    gpts=[openai_chatgpt, azure_chatgpt],
    allocation_function='max_headroom',
)
    
res = asyncio.run(chatgpt.chat_completion_list([ "your list of prompts" ]))
```


## How to Get API Keys
To obtain your OpenAI API key, visit the [official OpenAI site](https://platform.openai.com/account/api-keys). For Azure API key acquisition, please refer to the official Azure documentation.


