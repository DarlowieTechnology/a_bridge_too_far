from django.http import HttpResponse
from django.template import loader

from django.shortcuts import render
import genai_prices



# use this to display genAI pricing
#   
def index(request):
    """
    Target of HTTP GET from indexer/process.html.
    Displays API costs.
    
    Args:
        request

    Returns:
        render indexer/results.html

    """

    providers = [
        { "provider" : "anthropic", "model": "claude-haiku-4-5", "name": "Claude Haiku 4.5"  },
        { "provider" : "anthropic", "model": "claude-opus-4-5", "name": "Claude Opus 4.5"  },
        { "provider" : "anthropic", "model": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5"  },

        { "provider" : "avian", "model": "Meta-Llama-3.1-405B-Instruct", "name": "Llama 3.1 405B" },

        { "provider" : "azure", "model": "gpt-4", "name": "GPT-4" },
        { "provider" : "azure", "model": "phi-3-medium-128k-instruct", "name": "Phi-3 Medium 128K Instruct" },
        { "provider" : "azure", "model": "phi-4", "name": "Phi-4" },
        { "provider" : "azure", "model": "wizardlm-2-8x22b", "name": "WizardLM-2 8x22B" },

        { "provider" : "bedrock", "model": "openai.gpt-oss-120b-1:0", "name": "GPT-OSS 120B" },
        { "provider" : "bedrock", "model": "apac.anthropic.claude-sonnet-4-5-20250929-v1", "name": "Claude Sonnet 4.5" },

        { "provider" : "cerebras", "model": "gpt-oss-120b", "name": "GPT-OSS 120B" },
        { "provider" : "cerebras", "model": "llama-3.3-70b", "name": "Llama 3.3 70B" },

        { "provider" : "cohere", "model": "command-a", "name": "Command A" },
        { "provider" : "cohere", "model": "command-r", "name": "Command R" },
        { "provider" : "cohere", "model": "command-r-plus", "name": "Command R+" },

        { "provider" : "deepseek", "model": "deepseek-chat", "name": "DeepSeek Chat" },
        { "provider" : "deepseek", "model": "deepseek-reasoner", "name": "DeepSeek R" },

        { "provider" : "google", "model": "gemini-2.5-pro", "name": "Gemini 2.5 Pro" },
        { "provider" : "google", "model": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview" },

        { "provider" : "groq", "model": "llama-3.1-405b-reasoning", "name": "Llama 3.1 405B" },
        { "provider" : "groq", "model": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile 128k" },
        { "provider" : "groq", "model": "openai/gpt-oss-120b", "name": "GPT-OSS 120B" },
        { "provider" : "groq", "model": "qwen/qwen3-32b", "name": "Qwen3 32B 131k" },

        { "provider" : "openai", "model": "gpt-4o", "name": "GPT-4o" },
        { "provider" : "openai", "model": "gpt-5", "name": "GPT-5" },

        { "provider" : "openrouter", "model": "anthropic/claude-haiku-4.5", "name": "Claude Haiku 4.5" },
        { "provider" : "openrouter", "model": "anthropic/claude-opus-4.5", "name": "Claude Opus 4.5" },
        { "provider" : "openrouter", "model": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5" },
        { "provider" : "openrouter", "model": "chatgpt-4o-latest", "name": "GPT-4o" },
        { "provider" : "openrouter", "model": "cohere/command-a", "name": "Command A" },
        { "provider" : "openrouter", "model": "cohere/command-r", "name": "Command R" },
        { "provider" : "openrouter", "model": "cohere/command-r-plus", "name": "Command R+" },
        { "provider" : "openrouter", "model": "deepseek-chat", "name": "DeepSeek V3" },
        { "provider" : "openrouter", "model": "deepseek-r1", "name": "DeepSeek R1" },
        { "provider" : "openrouter", "model": "gemini-2.5-pro", "name": "Gemini 2.5 Pro" },
        { "provider" : "openrouter", "model": "gpt-4", "name": "GPT-4" },
        { "provider" : "openrouter", "model": "gpt-4o", "name": "GPT-4o" },
        { "provider" : "openrouter", "model": "openai/gpt-5-pro", "name": "GPT-5 Pro" },
        { "provider" : "openrouter", "model": "openai/gpt-oss-120b", "name": "GPT-OSS 120B" },
        { "provider" : "openrouter", "model": "grok-3", "name": "Grok-3" },
        { "provider" : "openrouter", "model": "llama-3.1-405b", "name": "Llama 3.1 405B (base)" },
        { "provider" : "openrouter", "model": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B Instruct" },
        { "provider" : "openrouter", "model": "microsoft/phi-3-medium-128k-instruct", "name": "Phi-3 Medium 128K Instruct" },
        { "provider" : "openrouter", "model": "microsoft/phi-4", "name": "microsoft/phi-4" },
        { "provider" : "openrouter", "model": "microsoft/wizardlm-2-8x22b", "name": "microsoft/wizardlm-2-8x22b" },
        { "provider" : "openrouter", "model": "qwen3-235b-a22b", "name": "Qwen3 235B A22B" },

        { "provider" : "x-ai", "model": "grok-3", "name": "Grok 3" },
        { "provider" : "x-ai", "model": "grok-4-0709", "name": "Grok 4" }
    ]

    genai_prices.update_prices.wait_prices_updated_sync(6.0)

    context = {}
    if "totalrequests" in request.GET:
        context["totalrequests"] = request.GET["totalrequests"]
    else:
        context["totalrequests"] = 0
    if "totalinputtokens" in request.GET:
        context["totalinputtokens"] = request.GET["totalinputtokens"]
    else:
        context["totalinputtokens"] = 0
    if "totaloutputtokens" in request.GET:
        context["totaloutputtokens"] = request.GET["totaloutputtokens"]
    else:
        context["totaloutputtokens"] = 0

    context["llminfo"] = []
    for providerInfo in providers:
        price_data = genai_prices.calc_price(
            genai_prices.Usage(input_tokens=int(context["totalinputtokens"]), output_tokens=int(context["totaloutputtokens"])),
            model_ref= providerInfo["model"],
            provider_id = providerInfo["provider"]
        )
        item = {}
        item["provider"] = providerInfo["provider"]
        item["model"] = providerInfo["model"]
        item["name"] = providerInfo["name"]
        item["costusd"] = f"{price_data.total_price:.4f}"

        context["llminfo"].append(item)

    template = loader.get_template("llmcosts/index.html")
    return HttpResponse(template.render(context, request))

    return render(request, "llmcosts/index.html", context)



