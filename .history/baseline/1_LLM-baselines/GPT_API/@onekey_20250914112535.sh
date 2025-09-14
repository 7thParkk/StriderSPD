HF_HUB_OFFLINE=1

python llm_online.py -u https://api.chatanywhere.tech/v1/ -k 'your_api_key' -m gpt-4o -d benchmark_Pseudos_new -t 16384
python llm_online.py -u https://api.chatanywhere.tech/v1/ -k 'your_api_key' -m gpt-5-mini -d benchmark_Pseudos_new -t 16384

    # python llm_online.py -u https://api.deepseek.com/v1 -k deepseek-key -m deepseek-reasoner -d pseudo_code -t 65536
    # python llm_online.py -u https://api.deepseek.com/v1 -k deepseek-key -m deepseek-reasoner -d assembly_code -t 65536
